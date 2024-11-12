#stick to this convention: https://peps.python.org/pep-0008/ 
import torch.nn as nn 
import torch
import copy
import torch.optim as opt
from torch.nn import MSELoss as mse
import os
import numpy as np
import random
import PyQt5 as qt
import pyqtgraph as pg
import pickle
import gc
import tracemalloc


from cobel.policy.greedy import EpsilonGreedy #this is from the new_features version
from memory_modules.sfma_memory import SFMAMemory
from memory_modules.memory_utils.metrics import Learnable_DR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

visualOutput = False

class SFMA_DQNAgent():
    class callbacks():
        def __init__(self, rl_parent, trial_begin_fun=None, trial_end_fun=None):
            # store the hosting class
            self.rl_parent = rl_parent
            # store the trial end callback function
            self.trial_begin_fun = trial_begin_fun
            # store the trial end callback function
            self.trial_end_fun = trial_end_fun

        def on_episode_begin(self, epoch, logs):
            if self.trial_begin_fun is not None:
                self.trial_begin_fun(self.rl_parent.current_trial - 1, self.rl_parent)

        def on_episode_end(self, epoch, logs):
            # update trial count
            self.rl_parent.current_trial += 1
            self.rl_parent.session_trial += 1
            # stop training after the maximum number of trials was reached
            if self.rl_parent.session_trial >= self.rl_parent.max_trials:
                self.rl_parent.step = self.rl_parent.max_steps + 1
            if self.trial_end_fun is not None:
                self.trial_end_fun(self.rl_parent.current_trial - 1, self.rl_parent, logs)

    def __init__(self, modules, external_mem, replay_type, num_replay, epsilon=0.1, gamma=0.95, online_learning=False,
                 with_replay=True, trial_begin_fun=None, trial_end_fun=None, one_hot = False):
        # store the Open AI Gym interface
        self.interfaceOAI = modules['rl_interface']
        self.number_of_states = modules['world'].numOfStates()



        self.one_hot = one_hot
        if self.one_hot:
            self.obs_dict = dict()
            i = 0
            for k in modules['world'].env.keys():
                if k in ['world_limits', 'walls_limits', 'perimeter_nodes']:
                    self.obs_dict.update({k:modules['world'].env[k]})
                else:
                    state_one_hot = torch.zeros(self.number_of_states).to(device)
                    state_one_hot[i] = 1.0
                    self.obs_dict.update({k:state_one_hot})
                    i += 1
        else:
            self.obs_dict = modules['world'].env.copy()
        self.all_states = torch.stack([torch.tensor(v) for v in list(self.obs_dict.values())[:-3]]).to(device)
        if not self.one_hot:
            self.all_states = torch.transpose(self.all_states,1,3)


        if self.one_hot: 
            self.observation_space = np.array(list(self.obs_dict.values())[0].cpu()).shape            
        else:
            self.observation_space = np.array(list(self.obs_dict.values())[0]).shape
        print("os: ", self.observation_space)
        if len(self.observation_space) == 1:   # the inputs are vectors
            self.vectorInput = True
        else:      # the inputs are images
            self.vectorInput = False

        # if the agent is also trained with the online experience
        self.online_learning = online_learning
        # if there is replay
        self.with_replay = with_replay
        self.batch_size = 32

        # the type of replay
        self.replay_type = replay_type
        # define the maximum number of steps
        self.max_steps = 10 ** 10
        # keeps track of current trial
        self.current_trial = 0  # trial count across all sessions (i.e. calls to the train/simulate method)
        self.session_trial = 0  # trial count in current seesion (i.e. current call to the train/simulate method)
        # define the maximum number of trials
        self.max_trials = 0
        ### Initializers for SFMA side
        # logging
        self.logs = {}
        self.logs['rewards'] = []
        self.logs['q_values'] = []
        self.logs['steps'] = []
        self.logs['behavior'] = []
        self.logs['modes'] = []
        self.logs['errors'] = []
        self.logs['replay_traces'] = {'start': [], 'end': []}
        self.logging_settings = {}
        for log in self.logs:
            self.logging_settings[log] = False
        # training
        self.online_replays_per_trial = num_replay[0]  # number of replay batches which start from the terminal state
        self.offline_replays_per_trial = num_replay[1]  # number of offline replay batches
        self.update_per_batch = 1 # number of updates for one single batch
        self.random_replay = False  # if true, random replay batches are sampled
        # retrieve number of actions
        self.number_of_actions = self.interfaceOAI.action_space.n
        self.build_model()
        self.target_model_update = 1e-2
        self.policy = EpsilonGreedy(epsilon)
        self.test_policy = self.policy

        self.memory = external_mem
        self.gamma = gamma
        self.lr = 0.0001
        self.compile(opt.Adam)
        self.engagedCallbacks = self.callbacks(self, trial_begin_fun, trial_end_fun)

        self.recent_observation = None
        self.recent_action = None


    def build_model(self):
        if not self.vectorInput:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels = self.observation_space[2], out_channels = 16, kernel_size=(8,8), stride = 4),
                nn.ReLU(),
                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (4,4), stride = 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2592 ,256),
                nn.ReLU(),
                nn.Linear(256,self.number_of_actions)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.observation_space[0] ,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,self.number_of_actions)
            )
        self.model = self.model.to(device)
        self.target_model = copy.deepcopy(self.model).to(device)

    def compile(self, optimizer):
        self.opt = optimizer(self.model.parameters(), lr = self.lr)
        self.target_opt = optimizer(self.target_model.parameters(), lr = self.lr)

    def train(self, number_of_trials=100, max_number_of_steps=50):
        self.training = True
        self.max_trials = number_of_trials
        self.session_trial = 0
        last_trial = False
        # record the total elapsed steps
        elapsed_steps = 0
        for trial in range(number_of_trials):
            # reset environment
            state = self.interfaceOAI.reset()
            #old_state = torch.tensor(state['observation'])
            # log cumulative reward
            trial_log = {'rewards': 0, 'steps': 0, 'modes': None, 'errors': 0, 'behavior': []}

            for step in range(max_number_of_steps):
                # determine next action
                # perform action
                action, _ = self.forward(state)
                next_state, reward, terminal, _ = self.interfaceOAI.step(action)

                stop_episode = terminal
                if step==max_number_of_steps-1:
                    stop_episode = True
                if trial == number_of_trials - 1:
                    last_trial = True
                self.backward(next_state, reward, terminal, stop_episode, last_trial)
                state = next_state
                # log behavior and reward
                #TODO undo comment
                trial_log['behavior'] += [[state['observationIdx'], action]]
                trial_log['rewards'] += reward
                if terminal:
                    break
            # log step and difference to optimal Q-function
            trial_log['steps'] = step
            elapsed_steps += step
            if trial % 100 == 0:

                print('%s/%s Episode step: %s   Elapsed step: %s' % (trial+1, number_of_trials, step, elapsed_steps))
                mem_current, mem_peak = tracemalloc.get_traced_memory()
                print(f"memory usage: current: {mem_current}, peak: {mem_peak}")
            # store trial logs
            #TODO undo comment
            for log in trial_log:
                if self.logging_settings[log]:
                    self.logs[log] += [trial_log[log]]
            # callback
            self.engagedCallbacks.on_episode_end(trial, {'episode_reward': trial_log['rewards'],
                                                         'nb_episode_steps': trial_log['steps'],
                                                         'nb_steps': elapsed_steps})

    def backward(self, next_state, reward, terminal, stop_episode, last_trial):
        # make experience with state idx, not the real state
        experience = {'state': self.recent_state['observationIdx'], 'action': self.recent_action, 'reward': reward,
                      'next_state': next_state['observationIdx'], 'terminal': terminal}

        # train the agent online if needed
        if self.online_learning:
            self.update_network([experience])

        # store experience in SFMA memory
        self.memory.store(experience)
        # this is only for random memory storage. From keras-rl:
        if stop_episode:
            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(next_state)

        if stop_episode and self.with_replay:
            #we save all q_values before the update
            self.save_q_values()
            # offline replay multiple times
            if self.replay_type == 'SR_SU':
                for _ in range(self.online_replays_per_trial):
                    self.update_sequential(self.batch_size, next_state['observationIdx'])
                for _ in range(self.offline_replays_per_trial):
                    self.update_sequential(self.batch_size)

            elif self.replay_type == 'SR_AU':
                for _ in range(self.online_replays_per_trial):
                    self.update_average(self.batch_size, next_state['observationIdx'])
                for _ in range(self.offline_replays_per_trial):
                    self.update_average(self.batch_size)
                
            if last_trial:
                # save q_values once again after last update
                self.save_q_values()


    def replay(self, replayBatchSize=200, state=None):
        # sample batch of experiencess
        if self.memory.beta == 0:
            self.random_replay = True
        if self.random_replay:
            mask = np.array(self.memory.C != 0)
            replayBatch = self.memory.retrieve_random_batch(replayBatchSize, mask, False)
        else:
            replayBatch = self.memory.replay(replayBatchSize, state)

        return replayBatch

    def update_average(self, replayBatchSize, currentState=None):
        replay_batch = self.replay(replayBatchSize, currentState)


        if self.logging_settings['replay_traces']:
            self.logs['replay_traces']['end'] += [replay_batch]
        for _ in range(self.update_per_batch):
            
            self.update_network(replay_batch)

    def update_local(self, replayBatchSize, currentState=None):
        '''
        Update Q network by averaging each replay batch
        '''
        for _ in range(self.replays_per_trial):
            replay_batch = self.replay(replayBatchSize, currentState)
            if self.logging_settings['replay_traces']:
                self.logs['replay_traces']['end'] += [replay_batch]
            self.update_network(replay_batch, local_target=True)

    def update_sequential(self, replayBatchSize, currentState=None):
        replay_batch = self.replay(replayBatchSize, currentState)
        if self.logging_settings['replay_traces']:
            self.logs['replay_traces']['end'] += [replay_batch]
        for _ in range(self.update_per_batch):
            for item in replay_batch:
                self.update_network([item])

    def update_sequential_average(self, replayBatchSize, currentState=None):
        replay_batches = []
        minimum_batch_size = replayBatchSize
        for _ in range(self.replays_per_trial):
            replay_batch = self.replay(replayBatchSize, currentState)
            if self.logging_settings['replay_traces']:
                self.logs['replay_traces']['end'] += [replay_batch]
            replay_batches.append(replay_batch)
            minimum_batch_size = min(minimum_batch_size, len(replay_batch))
        for step in range(minimum_batch_size):
            experiences = []
            for batch in replay_batches:
                experiences.append(batch[step])
            self.update_network(experiences)

    def update_network(self, experiencebatch, local_target=False):
        # prepare placeholders for updating
        replay_size = len(experiencebatch)
        state0_batch, reward_batch, action_batch, terminal1_batch, state1_batch = [], [], [], [], []
        state0Idx_batch = []
        state1Idx_batch = []
        for e in experiencebatch:
            state0Idx_batch.append(e['state'])
            state1Idx_batch.append(e['next_state'])
            state0_batch.append(self.Idx2Observation(e['state']))
            state1_batch.append(self.Idx2Observation(e['next_state']))
            reward_batch.append(e['reward'])
            action_batch.append(e['action'])
            terminal1_batch.append(0 if e['terminal'] else 1)
        # Prepare parameters.

        if self.one_hot:
            state0_batch = torch.stack(state0_batch)
            state1_batch = torch.stack(state1_batch)
            terminal1_batch, reward_batch = torch.tensor(terminal1_batch).to(device), torch.tensor(reward_batch).to(device)
        else: 
            state0_batch, state1_batch = torch.tensor(np.array(state0_batch)).to(device), torch.tensor(np.array(state1_batch)).to(device)
            terminal1_batch, reward_batch = torch.tensor(terminal1_batch).to(device), torch.tensor(reward_batch).to(device)
            # we first initialize the target values the same as the predictions over the current states, and then compute the real target values,
            # assign it to the Q value of the selected action. In this way, all other Q values got cancelled out.
            state0_batch = torch.transpose(state0_batch, 1, 3).to(device)
            state1_batch = torch.transpose(state1_batch, 1, 3).to(device)


        q_targets = self.model(state0_batch).detach().clone() 

        # this is the prediction (before model update)

        if not local_target:
            # get q values from the target network
            target_q_values = self.target_model(state1_batch) 
            # these are the targets values
            q_batch = torch.max(target_q_values.detach(), dim=1)[0]  
            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            Rs = reward_batch + discounted_reward_batch
        else:
            # we compute Q targets locally to propagate the reward infomation
            observations = self.get_all_observations()
            # observations = np.eye(self.number_of_states)
            Q_local = self.target_model(observations) 
            Rs = []
            for e in experiencebatch:
                # compute target
                target = e['reward']
                target += self.gamma * e['terminal'] * torch.amax(Q_local[e['next_state']]) 
                # update local Q-function
                Q_local[e['state']][e['action']] = target
                Rs.append(target)
            Rs = torch.tensor(Rs)        
        Rs = Rs.clone().detach().to(device)
        # prepare varibles for updating
        for idx, (q_target, R, action) in enumerate(zip(q_targets, Rs, action_batch)):
            q_target[action] = R  # update action with estimated accumulated reward
        # update the model

        metrics = self.update_model(state0_batch, q_targets)

        return metrics

    def update_model(self, observations, targets, number_of_updates=1):
        '''
        At the end of each episode, his function updates the model on a batch of experiences as well as the target model 

        | **Args**
        | observations:                 The observations.
        | targets:                      The targets.
        | number_of_updates:            The number of backpropagation updates that should be performed on this batch.
        '''
        # update online model
        for update in range(number_of_updates):
            metrics = self.train_on_batch(observations, targets)
        # update target model by blending it with the online model
        state_dict_target = self.target_model.state_dict()
        state_dict_online = self.model.state_dict()

        weights_target = {k: v.clone().detach() for k, v in state_dict_target.items()}
        weights_online = {k: v.clone().detach() for k, v in state_dict_online.items()}

        for k in weights_target.keys():
            weights_target[k] += self.target_model_update * (weights_online[k] - weights_target[k])

        updated_state_dict = {k: v.clone().detach() for k, v in weights_target.items()}

        self.target_model.load_state_dict(updated_state_dict)


        return metrics

    def Idx2Observation(self, ObservationIdx):
        '''
        This function convert a list of state indice to real states by querying the world module
        '''
        obsIdxList = list(self.obs_dict.keys())
        obsKey = obsIdxList[ObservationIdx]
        
        return self.obs_dict[obsKey]
        


    def get_all_observations(self):
        observations = list(self.obs_dict.values())[:-3]
        return torch.tensor(np.asarray(observations)).to(device)  # convert to tensor and move to device

    def forward(self, observation):
        # Select an action based on the state oberservation (ususally an image)
        state_idx = observation['observationIdx']
        state = self.Idx2Observation(state_idx)
        #convert to transposed tensor
        if not self.vectorInput:
            state = torch.transpose(torch.tensor(state), 0, 2).unsqueeze(0).to(device)

        q_values = self.model(state).flatten()

        if self.training:
            action = self.policy.select_action(v=q_values.cpu().detach())
        else:
            action = self.test_policy.select_action(v=q_values.cpu())

        self.recent_state = observation
        self.recent_action = action

        return action, q_values
    
    def train_on_batch(self, observations, targets):
        '''
        This function updates the online model 
        '''
        mse = nn.MSELoss()

        self.model.train()
        self.opt.zero_grad()
        pred = self.model(observations.to(device))
        loss = mse(pred, targets)
        loss.backward()
        self.opt.step()
        return loss
    
    def save_q_values(self):
        q_values = self.model(self.all_states).detach().cpu().numpy()

        assert q_values.shape[0] == self.number_of_states
        #TODO undo comment
        self.logs['q_values'].append(q_values)


