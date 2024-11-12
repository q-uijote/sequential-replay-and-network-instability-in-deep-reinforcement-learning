import torch
import torch.nn as nn 
from torch.nn import MSELoss as mse
import copy
import torch.optim as opt
import os
import numpy as np
import random
import PyQt5 as qt
import pyqtgraph as pg
import pickle
import gc
from deep_sfma_torch import SFMA_DQNAgent
import time
import sys
from optimal_Q import optimal_q_values
import gym

from spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from observations.image_observations import ImageObservationUnity
from analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor
from interfaces.oai_gym_interface import OAIGymInterface
from frontends.frontends_unity import FrontendUnityOfflineInterface
from sfma_memory_adapted import SFMAMemory
from memory_modules.memory_utils.metrics import Learnable_DR

import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import itertools

import tracemalloc
import psutil
import threading

visualOutput = False

# reward function
def rewardCallback(values):
    # the standard reward for each step taken is negative, making the agent seek short routes
    reward = 0.0
    stopEpisode = False
    if values['currentNode'].goalNode:
        reward = 1.0
        stopEpisode = True

    return reward, stopEpisode

def trialEndCallback(trial, rlAgent, logs):
    if visualOutput:
        # update the visual elements if required
        rlAgent.performanceMonitor.update(trial, logs)
        if hasattr(qt.QtGui, 'QApplication'):
            qt.QtGui.QApplication.instance().processEvents()
        else:
            qt.QtWidgets.QApplication.instance().processEvents()



def single_run(visual_output, running_env, replay_type, step_size, batch_size, num_replay, beta, epoch, data_folder, one_hot, action_space, num_random_replays=0):
    data_folder = data_folder + '/action_space_%s_one_hot_%s' % (action_space, one_hot)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)


    #### choose and load the environment
    global visualOutput
    visualOutput = visual_output 
    start_goal_nodes = {'TunnelMaze_LV1': [[1], [14]], 'TunnelMaze_LV2': [[6], [32]], 'TunnelMaze_LV3': [[21], [60]],
                        'TunnelMaze_LV4': [[42], [94]], 'TunnelMaze_LV4_hc': [[204], [458]]}
    # this is the main window for visual outputs
    mainWindow=None
    # if visual output is required, activate an output window
    if visual_output:
        mainWindow = pg.GraphicsWindow(title="Unity Environment Plot")
        layout = pg.GraphicsLayout(border=(30, 30, 30))
        mainWindow.setCentralItem(layout)

    # determine world info file path
    worldInfo = os.path.dirname(os.path.abspath(__file__)) + '/../EM_Spatial_Learning/environments_unity/offline_unity/%s_ss%s_Infos.pickle' % (running_env, step_size)

    # a dictionary that contains all employed modules
    modules = dict()


    def get_modified_action_space(self):
        return gym.spaces.Discrete(action_space)


    Four_Connected_Graph_Rotation.get_action_space = get_modified_action_space

    modules['world'] = FrontendUnityOfflineInterface(worldInfo)
    modules['observation'] = ImageObservationUnity(modules['world'], mainWindow, visualOutput)
    modules['spatial_representation'] = Four_Connected_Graph_Rotation(modules, {'startNodes':start_goal_nodes[running_env][0], 'goalNodes':start_goal_nodes[running_env][1], 'start_ori': 90, 'cliqueSize':4}, step_size=step_size)
    modules['spatial_representation'].store_topology('%s.pickle'%running_env)
    modules['spatial_representation'].set_visual_debugging(visualOutput, mainWindow)
    modules['rl_interface']=OAIGymInterface(modules, visualOutput, rewardCallback, withStateIdx=True)


    #### load the memory replay module, one is SMA and the other Random
    numberOfActions = modules['rl_interface'].action_space.n

    numberOfStates = modules['world'].numOfStates()
    gammaDR = 0.2
    ## for the SMA, load the DR matrix for the env
    ## if there is no stored DR, start a simple simulation where an agent randomly explore the env and update DR incrementally, then store it
    DR_metric = Learnable_DR(numberOfStates, gammaDR)
    DR_path = os.path.dirname(os.path.abspath(__file__)) + '/../EM_Spatial_Learning/memory_modules/stored/DR_%s_%s_ss%s_gamma_%s.pickle' % (running_env, numberOfActions, step_size, gammaDR)

    ifPretrained = DR_metric.loadD(DR_path)
    print("ifPretrained = ", ifPretrained)
    if not ifPretrained:
        modules['spatial_representation'].generate_behavior_from_action('reset')
        for i in range(200000):
            print(i)
            currentStateIdx = modules['world'].envData['imageIdx']
            action = np.random.randint(low=0, high=numberOfActions)
            modules['spatial_representation'].generate_behavior_from_action(action)
            nextStateIdx = modules['world'].envData['imageIdx']
            # update DR matrix with one-step experience
            DR_metric.updateD(currentStateIdx, nextStateIdx, lr=0.1)
        # store the learned DR for this env
        DR_metric.storeD(DR_path)
        print("DR_metric stored in %s" % (DR_path)) 
    # initialize hippocampal memory
    HM = SFMAMemory(numberOfStates, numberOfActions, DR_metric, extend_by=num_random_replays)
    # load the agent
    epsilon = 0.1
    rlAgent = SFMA_DQNAgent(modules, HM, replay_type, num_replay, epsilon, 0.95, with_replay=True, online_learning=False,
                            trial_begin_fun=None, trial_end_fun=trialEndCallback, one_hot = one_hot)
    # number of online replays starting from the terminal state
    rlAgent.online_replays_per_trial = num_replay[0]
    # number of offline replays
    rlAgent.offline_replays_per_trial = num_replay[1]
    rlAgent.batch_size = batch_size

    # common settings
    rlAgent.memory.beta = beta
    rlAgent.memory.mode = 'reverse'
    rlAgent.memory.reward_mod = True
    rlAgent.memory.reward_modulation = 1.0
    rlAgent.logging_settings['steps'] = True #TODO set to True
    rlAgent.logging_settings['replay_traces'] = False #TODO set to True

    # set the performance monitor
    perfMon=UnityPerformanceMonitor(rlAgent,mainWindow,visualOutput)
    rlAgent.performanceMonitor=perfMon
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rlAgent=rlAgent
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rlAgent=rlAgent

    
    #CALCULATION OF OPTIMAL Q VALUES
    path_optimal_q = data_folder + '/%s_optimal_Q.pickle' % (running_env)


    if not os.path.exists(path_optimal_q):

        states = list(modules['world'].env.keys())[:-3]  # last three keys are no states

        angles = [0,90,-180,-90]
        positions = [(modules['spatial_representation'].nodes[state // 4].x, modules['spatial_representation'].nodes[state // 4].y, angles[state%4]) for state in range(numberOfStates)]
        neighbors = [[n.index for n in modules['spatial_representation'].nodes[state // 4].neighbors  if n.index >= 0] for state in range(numberOfStates)]

        state_dict = {'states' : positions, 'neighbors' : neighbors}
        
        optimal_q = optimal_q_values(state_dict, start_goal_nodes[running_env][1][0]*4, numberOfStates, numberOfActions, 0.95)

        with open(path_optimal_q, 'wb') as handle:
            pickle.dump(optimal_q, handle)
        print(np.max(optimal_q))


    # start the training
    rlAgent.train(number_of_trials=500, max_number_of_steps=600)
    # end the training
    modules['world'].stopUnity()
    if mainWindow is not None:
        mainWindow.close()
    # extract replayed state index in each batch
    replayed_history = []

    for batch in rlAgent.logs['replay_traces']['end']:
        statebatch = [[e['state'], e['action'], e['reward'], e['next_state']] for e in batch]
        replayed_history.append(statebatch)

    ifanalyze = True
    if ifanalyze:
        data_dir = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # store the trajectories in all training trials
        data_path = data_dir + '/TrainingTrajs_%s_%s_%s.pickle' % (replay_type, running_env, epoch)

        if not os.path.exists(data_path):

            with open(data_path, 'wb') as handle:
                pickle.dump(modules['spatial_representation'].trajectories, handle)

        data_path = data_dir + '/ReplayBatches_%s_%s_%s.pickle' % (replay_type, running_env, epoch)
        if not os.path.exists(data_path):

            with open(data_path, 'wb') as handle:
                pickle.dump(replayed_history, handle)
        
        data_path = data_dir + '/Q_values_%s_%s_%s.pickle' % (replay_type, running_env, epoch)

        if not os.path.exists(data_path):
            with open(data_path, 'wb') as handle:
                pickle.dump(rlAgent.logs['q_values'], handle)

def monitor_memory_usage(interval=15):
    """ Monitor memory usage of all processes in the pool. """
    current_process = psutil.Process(os.getpid())
    
    while True:
        children = current_process.children(recursive=True)  # Get all child processes
        total_memory = sum([child.memory_info().rss for child in children]) / (1024 * 1024)  # Convert to MB
        
        print(f"Memory Usage: {total_memory:.2f} MB")
        time.sleep(interval)  # Wait for the next check
 

def run_simulation(params):

    print(f"Start simulation with params: \n{params}")
    print("visual_output, running_env, replay_type, step_size, batch_size, num_replay, beta, epoch, data_folder, one_hot")

    single_run(*params)
    print()
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    print(f"Single run finished with memory usage: current: {mem_current}, peak: {mem_peak}")

if __name__ == "__main__":

    tracemalloc.start()

    running_env = ['TunnelMaze_LV4']
    visual_output = [False]
    step_size = [1.0]
    betas = [10]#[0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10]
    epochs = range(5)
    replay_type = ['SR_AU']
    batch_size = [32]
    num_replay = [[50, 0]]#[[10,0],[20, 0],[50, 0]]
    one_hot_encoding = [False]
    action_space = [3] # 3 = only left turn, right turn, forward movement permitted
    num_random_replays = [0] # if this is != 0, SFMA batches are enriched by a number of randomly sampled experiences
    data_folder = 'data/sequential_replay_1_torch'
    # number of parallel processes
    if len(sys.argv) > 1:
        num_p = int(sys.argv[1])
        if int(sys.argv[1]) > 1:
            monitor_thread = threading.Thread(target=monitor_memory_usage, args=(1,), daemon=True)
            monitor_thread.start()
    else:
        num_p = 1

    # extend path e.g. for google colab or HPC file system
    if len(sys.argv) > 2:
        data_folder = sys.argv[2] + data_folder 
    # start thread for monitoring memory usage

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    print(data_folder)

    print(f"CPU available: {mp.cpu_count()}")
    print(f"CUDA count: {torch.cuda.device_count()}")
    print(f"Executing with {num_p} processes")

    params = list(itertools.product(visual_output,running_env, replay_type, step_size, batch_size,num_replay, betas, epochs, [data_folder],one_hot_encoding, action_space,num_random_replays))

    print("Num of params to process: ", len(params))

    start_total = time.time()
    mp.set_start_method('spawn') #CUDA runtime does not support fork methodq_values

    exe_times = []
    
    if num_p > 1:

        with Pool(num_p) as p:
            p.map(run_simulation, params)
        
    else:
        for i, param in enumerate(params):
            start_time = time.time()
            run_simulation(param)
            exe_time = time.time() - start_time
            exe_times.append(exe_time)
            print(f"Execution time of epoch {i}: {exe_time}")
    for i, exe_time in enumerate(exe_times):
        print(f"Execution time of run number {i}: {exe_time}")
    print(f"Total execution time: {time.time() - start_total}")
    tracemalloc.stop()

