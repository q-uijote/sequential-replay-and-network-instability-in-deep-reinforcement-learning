import os, re
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mat
import matplotlib.cm as cm
from itertools import product
import numpy 
from matplotlib.cm import ScalarMappable
import time
import tracemalloc # track memory allocation
from numpy import newaxis as na
import seaborn as sns
from matplotlib.gridspec import GridSpec


# Switch to Agg backend for headless environments
#matplotlib.use('Agg')

## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}

fontsize_legend = 'small'
fontsize = 14
labelsize = 12

mat.rc('font', **font)
## env and agent information
start_goal_nodes = {'TunnelMaze_LV1': [[1], [14]], 'TunnelMaze_LV2': [[6], [32]], 'TunnelMaze_LV3': [[21], [60]],
                    'TunnelMaze_LV4': [[42], [94]], 'TunnelMaze_New': [[44], [101]], 'TMaze': [[3], [11]]}

project_folder = os.path.dirname(os.path.abspath(__file__)) + '/../EM_Spatial_Learning'
batch_size = 32

#colors = cm.get_cmap('Set1', 5)  # Tab10 has 10 distinct colors
colors = mat.colormaps.get_cmap('Set1')
betas = [0,1,2,5,10]
colors = {beta : colors(idx) for idx, beta in enumerate(betas)}

def draw_envs(running_env, env_top_path, axs):
    with open(env_top_path, 'rb') as handle:
        data = pickle.load(handle)
    # here each node is a XY coordinate and each edge is a list containing 2 integers, indicating a connectivity
    # between 2 nodes
    
    nodes = data['nodes']
    edges = data['edges']
    walls = data['walls']

    # draw the walls
    for w in walls:
        xy = w[:2]
        wh = w[2:] - w[:2]
        rect = mat.patches.Rectangle(xy, wh[0], wh[1], edgecolor='k', facecolor='none')
        axs.add_patch(rect)
    # draw the edges
    for e in edges:
        axs.plot(nodes[e][:, 0], nodes[e][:, 1], color='k', zorder=1)
    # draw the nodes
    axs.scatter(nodes[:, 0], nodes[:, 1], facecolors='white', edgecolors='b', s=80, zorder=3)
    # color the start and end nodes
    special = start_goal_nodes[running_env]
    axs.scatter(nodes[special[0][0]][0], nodes[special[0][0]][1], color='blue', s=70, zorder=3)
    axs.scatter(nodes[special[1][0]][0], nodes[special[1][0]][1], color='green', s=70, zorder=3)
    return nodes

def arrow_info(traj, nodes):
    # draw the arrows that define the trajectory
    arrow_pos = []
    # the X and Y components for each arrow direction
    U = []
    V = []
    for n in traj:
        # define the start and end of the arrow
        arrow_pos.append(nodes[n[0]])
        orientation = n[1]
        if orientation == 0:
            U.append(1)
            V.append(0)
        elif orientation == 90:
            U.append(0)
            V.append(1)
        elif orientation == -90:
            U.append(0)
            V.append(-1)
        elif orientation == -180:
            U.append(-1)
            V.append(0)

    arrow_pos = np.array(arrow_pos)
    return arrow_pos, U, V




def visualize_LC(data, handles, beta, running_env, action_space, num_replays, x_label, y_label, show_mean = True, show_std = False, show_all = False, plot_name = None, show_legend = False):
    """ 
    Input: data of size (trials, episodes, num_states), axs (where we plot everything)
    This function plots the Mean over all (s,a) tuples
    together with standard devitations
    """
    fig, axs = handles
    fig.set_size_inches(6.4, 4.8)

    # shape is (trials, episodes)
    mean = np.mean(data, axis = 0)

    num_trials = data.shape[0]
    num_episodes = data.shape[1]

    # MSVE(Q, Q_optimal) ist plotted as line graph
    axs.plot(data.mean(axis = 0), linewidth = 1, color = colors[beta], label = 'beta=%s'% (beta))
    
    png_name = 'LC_mean_%s_as%s_oh%s_r%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0], params['runs'])

    if show_std:
        axs.fill_between(range(num_episodes), data.mean(axis = 0) - data.std(axis = 0), data.mean(axis=0) + data.std(axis=0), color = colors[beta], alpha=0.5)
        png_name = 'LC_std_%s_as%s_oh%s_r%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0],params['runs'])

    if show_all:
        axs.plot(data.T, linewidth = 1, color = colors[beta], alpha = 0.6, label = 'beta = %s' % beta)
        png_name = 'LC_all_%s_as%s_oh%s_r%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0], params['runs'])
    axs.axhline(y=0, color='black', linewidth=1)
    axs.set_xticks(np.arange(0, num_episodes + 1, 100))
    axs.xaxis.grid(True, which='both')  # Enable grid for x-axis
    axs.yaxis.grid(True, which='both')  # Enable grid for y-axis
    axs.set_xlabel(x_label, fontsize=fontsize)
    axs.set_ylabel(y_label, fontsize=fontsize)    
    axs.tick_params(axis='both', which='major', labelsize=labelsize)
    if show_legend:
        axs.legend(fontsize=fontsize_legend)

    plt.tight_layout()
    if plot_name:
        png_name = plot_name

    if params['savefigs']:
        setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
        path = 'plots/' + setup
        if not os.path.exists(path):
            os.makedirs(path)
        beta = 0
        png_name = path + png_name
        print("saving fig as: ", png_name)
        fig.savefig(png_name, dpi=300,bbox_inches='tight')
    


def analyse_Matrix(A, axes=None):
    print("ANALYSE MATRIX: ", A)
    print("Shape: ", A.shape)
    print("min,max: ", np.min(A), np.max(A))
    print("0.1, 0.9 percentile: ", np.percentile(A,10),  np.percentile(A, 90))
    if isinstance(axes,tuple):
        mean = np.mean(A,axis = axes)
        print(f"Mean over axis {axes}", mean)
        print("Shape: ", mean.shape)
        print("min,max: ", np.min(mean), np.max(mean))
        print("0.1, 0.9 percentile: ", np.percentile(mean,10),  np.percentile(mean, 90))
        
""" 
What exactly is the problem with CI_s,a calculation?

The closer a state is to the goal state, the greater is its q* value.

Especially at the beginning of learning, error w.r.t. q* is greater for these states

Intention:
Showing that CI is greater for (s,a)-tuples that are underrepresented in replay

1) Plot frequency of replay for each tuple
2) Plot CI_s,a but adjust for 

- what does CI_s,a look like for single episodes?
- what happens if i normalize CI_s,a score (Z-scores)? -> hides all effects

Problem:
- CI is generally greater for states closer to goal state
- why? the reward is more present in these states. thus, when a learned q(s,a) is interfered with, and information is lost.
"""

def visualize_msve(data_folder, running_env, params, handles, betas, r_type, num_replay, num_of_actions, num_of_states):
    
    #data_path_optimal_q = data_folder + '/%s_optimal_Q_action_space=%s.pickle' % (running_env, num_of_actions) 
    data_path_optimal_q = data_folder + '/%s_optimal_Q.pickle' % (running_env) 

    # load and process optimal q-function
    with open(data_path_optimal_q, 'rb') as handle:
        q_optimal = pickle.load(handle)
        # we dont consider nan values    
        mask = ~np.isnan(q_optimal)
        #q_optimal = q_optimal[mask]
        #masked = True

    fig, axs = handles
    
    colors = ['blue', 'red','yellow','green','orange']
    #markers = ['o','v','s']


    # if we calculate new values, we set this flag
    new_values = False

    
    # plot CI for each beta
    for idx, beta in enumerate(betas):

        # for faster processing, we store calculated values
        data_path_ci_data = data_folder + '/ci_data_beta_%s.pickle' % beta
        if not os.path.exists(data_path_ci_data):
            ci_data = {'%s/%s+%s/%s_%s' % (beta, num_replay[0], num_replay[1], r_type, env): [[[],[]] for epoch in params['epochs']] for beta, num_replay, r_type, env in product(betas, num_replays, r_types, envs)}
            with open(data_path_ci_data, 'wb') as handle:
                pickle.dump(ci_data, handle)
        else: 
            with open(data_path_ci_data, 'rb') as handle:
                ci_data = pickle.load(handle) 

        data_file = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
        
        data_beta = [[],[]]
        ci_mean_beta = [] 
        ci_episodes = []

        epoch_count = 0

        num_of_episodes = 500

        # we store all q_values here, then average, measure variance and ci later
        beta_all_q = []

        # average over all epochs
        for epoch in params['epochs']:

            #data_path_q_values = data_file + '/Q_values_%s_%s_%s_%s.pickle' % (r_type, running_env, num_of_episodes, epoch)
            data_path_q_values = data_file + '/Q_values_%s_%s_%s.pickle' % (r_type, running_env, epoch)
            data_path_replay = data_file + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, epoch)
            
            # only for epochs for which data has been generated
            if os.path.exists(data_path_q_values) and os.path.exists(data_path_replay):

                epoch_count += 1 

                #print("Beta: %s Epoch: %s" % (beta, epoch))
                all_msve = ci_data['%s/%s+%s/%s_%s'%(beta, num_replay[0], num_replay[1], r_type, running_env)][epoch][0]
                ci_episode = ci_data['%s/%s+%s/%s_%s'%(beta, num_replay[0], num_replay[1], r_type, running_env)][epoch][1]
                
                # if ci_episode was not calculated already
                if True: # not (ci_episode and all_msve): 

                    # set flag so we store new values later
                    new_values = True
                
                    # mean-squared value error for each episode 
                    all_msve = []

                    # load stored q-functions (all episodes) for current epoch
                    with open(data_path_q_values, 'rb') as handle:
                        q_functions = pickle.load(handle)

                    # for each episode, q_function was stored once
                    for q_function in q_functions:

                        # Create a one-hot encoded array of actions with maximum q-value
                        #max_q_function = np.argmax(q_function, axis=1)
                        #one_hot_q_function = np.zeros((num_of_states, num_of_actions))
                        #one_hot_q_function[np.arange(num_of_states), max_q_function] = 1

                        #print("VALUES = ", one_hot_q_function, one_hot_q_optimal) #remove
                        #print("q_function masked = ", q_function[mask], q_function[mask].shape)
                        diff = (q_function - q_optimal)**2


                        # consider only valid (s,a) pairs by applying mask
                        msve = diff[mask].mean()
                        # msve is msve of one hot encodings of actions with maximum q-values
                        # a q-function is optimal, if the policy (in our case greedy) behaves optimal
                        # msve = np.mean((one_hot_q_function - one_hot_q_optimal)**2)

                        all_msve.append(msve)

                    for i in range(len(all_msve) - 1):
                        msve_t1 = all_msve[i]
                        msve_t2 = all_msve[i+1]

                        # negative msve_diff indicates improvement of q-function
                        # positive msve_diff indicates interference 
                        msve_diff =  msve_t2 - msve_t1

                        # yields one msve_diff score per update
                        ci_episode.append(msve_diff)
                    
                    
                    # update ci_data so we can rewrite the pickle file later
                    ci_data['%s/%s+%s/%s_%s'%(beta, num_replay[0], num_replay[1], r_type, running_env)][epoch][0] = all_msve 
                    ci_data['%s/%s+%s/%s_%s'%(beta, num_replay[0], num_replay[1], r_type, running_env)][epoch][1] = ci_episode 
                

                data_beta[0].append(all_msve)
                data_beta[1].append(ci_episode)

            # data has not been generated
            else: 
                print("paths dont exits: ", data_path_q_values, data_path_replay)

        # The following block is experimental: I'm calculating mean first and then variance and ci
        # Mean CI over all epochs of one beta
        var_mean_beta = np.array(data_beta[0],dtype = float).mean(axis = 0)
        ci_mean_beta = np.array(data_beta[1],dtype = float).mean(axis = 0)


        if new_values:
            with open(data_path_ci_data, 'wb') as handle:
                pickle.dump(ci_data, handle)

        x_values = np.arange(num_of_episodes)
        x_values = np.arange(len(ci_mean_beta))

        num_betas = len(betas)
        bar_width = 1.0 / (num_betas)  # Adjust bar width so all betas fit next to each other
        bar_offset = bar_width * np.arange(num_betas)  # Offset for each beta
        

        # MSVE(Q, Q_optimal) ist plotted as line graph
        axs.plot(range(len(var_mean_beta)), var_mean_beta, linewidth = 1,color = colors[idx],
              label = 'beta=%s (avg over %s runs)'% (beta, epoch_count))
    

    axs.axhline(y=0, color='black', linewidth=1)
    axs.set_xticks(np.arange(0, 501 , 100))
    axs.xaxis.grid(True, which='both')  # Enable grid for x-axis
    axs.yaxis.grid(True, which='both')  # Enable grid for y-axis
    axs.set_ylabel('MSVE(Q,Q_optimal)', fontsize=fontsize)
    axs.set_xlabel('Episode Number', fontsize=fontsize)
    
    
    #axs.set_title('Error w.r.t. optimal Q function')
    axs.tick_params(axis='both', which='major', labelsize=labelsize)
    #axs.text(0.5, -0.2, 'Error w.r.t. optimal Q function', ha='center', fontsize=8, transform=axs.transAxes)



    plt.tight_layout()

    plt.savefig('MSVE_%s_%s_%s.png' % (running_env, action_space, num_replay), dpi=300, bbox_inches='tight')

def get_CI_sa(q,q_optimal, action, episode = -1):
    """
    This function returns CI per state for a specific action and either a specified episode 
    or for all episodes (default) 
    """
    num_trials = q.shape[0]
    num_updates = q.shape[1] - 1
    num_states = q.shape[2]
    action_index = action - 1 
    print("q_optimal = ", q_optimal)
    # mean squared error
    mse = np.square(q-q_optimal)[:,:,:,action_index]

    ci = np.zeros((num_trials, num_updates, num_states))

    for i in range(num_updates):
        ci[:,i] = np.maximum(mse[:,i+1] - mse[:,i], 0)
    
    # get mean over all episodes per trial
    # should be trials, 1, states
    if episode >= 0:
        ci = np.mean(ci[:,episode,:], axis = 0)
    return ci


def prepare_sa_plot(running_env, axs):
    project_folder = os.path.dirname(os.path.abspath(__file__)) + '/../EM_Spatial_Learning'
    top_path = project_folder + '/environments_unity/offline_unity/%s_ss%s_Top.pickle' % (running_env, 1.0)
    info_path = project_folder + '/environments_unity/offline_unity/%s_ss%s_Infos.pickle' % (running_env, 1.0)
    # load the env topology and visualize it
    nodes_positions = draw_envs(running_env, top_path, axs)
    with open(info_path, 'rb') as handle:
        data = pickle.load(handle)
    state_list = list(data.keys())[:-3]
    states = []
    for state in state_list:
        state = re.sub("[\[\]]","",state)
        state = state.split(",")
        states.append([int(state[0]),int(state[1])])
    arrow_pos, U, V = arrow_info(states, nodes_positions)
    return arrow_pos,  U, V

def plot_states(fig, axs, cmap, norm, values, params,vmin = 0,vmax = 1):
    """
    This function creates a plot that shows each state with a normalized value assigned to it. 
    Input data should not be normalized
    """
    running_env = params['env']
    project_folder = os.path.dirname(os.path.abspath(__file__)) + '/../EM_Spatial_Learning'
    top_path = project_folder + '/environments_unity/offline_unity/%s_ss%s_Top.pickle' % (running_env, 1.0)
    info_path = project_folder + '/environments_unity/offline_unity/%s_ss%s_Infos.pickle' % (running_env, 1.0)
    # load the env topology and visualize it
    nodes_positions = draw_envs(running_env, top_path, axs)
    with open(info_path, 'rb') as handle:
        data = pickle.load(handle)
    state_list = list(data.keys())[:-3]
    states = []
    for state in state_list:
        state = re.sub("[\[\]]","",state)
        state = state.split(",")
        states.append([int(state[0]),int(state[1])])
    arrow_pos, U, V = arrow_info(states, nodes_positions)
    #color_steps = np.linspace(vmin, vmax, len(arrow_pos))

    normalized_values = norm(values)

    width = 0.1
    length = 0.2
    # Create a ScalarMappable and add a colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)

    sm.set_array([])  # Required for colorbar
    cb = fig.colorbar(sm, ax=axs)
    cb.set_ticks(np.linspace(vmin, vmax, num=5)) 
    for xy, dx, dy, t in zip(arrow_pos, U, V, normalized_values):
        axs.arrow(xy[0], xy[1], dx * length, dy * length, width=width, head_width=3.5 * width,
                head_length=3 * width, color=cmap(t), zorder=2)
    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks


def draw_norm_sa(fig, axs, cmap, norm,  arrow_pos, U, V, normalized_values):
    # normalizing ci values
    #ci_mean_state = norm(ci_mean_state)
    # define the size of the arrow
    width = 0.1
    length = 0.2
    # Create a ScalarMappable and add a colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for colorbar
    fig.colorbar(sm, ax=axs)
    for xy, dx, dy, t in zip(arrow_pos, U, V, normalized_values):
        axs.arrow(xy[0], xy[1], dx * length, dy * length, width=width, head_width=3.5 * width,
                head_length=3 * width, color=cmap(t), zorder=2)

def replay_count(beta, action, data_folder, num_states, params, episodes = (0,500)):
    """ 
    This function returns the replay frequency for each state for the specified action averaged over all episodes
    """
    num_runs = params['runs']
    num_replay = params['num_replay']
    Replay_matrix = np.empty((500*num_replay[0],32,4),dtype='float64')
    data_file = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
    replayed_count = np.zeros((num_states), dtype='float64')
    for run in range(params['runs']):
        data_path_replay = data_file + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, run)
        assert os.path.exists(data_path_replay), f"Data path '{data_path_replay}' does not exist"
        # load replay_batches
        with open(data_path_replay, 'rb') as handle:
            print("Loading... Replay_matrix.shape: ", Replay_matrix.shape)
            # 25000 x 32 x 4
            Replay_matrix[:] = pickle.load(handle)#replay_batches
        Replays = Replay_matrix.reshape(500, num_replay[0]*32, -1)

        # TODO: select only episodes of interest

        # shape 500 x 1600
        replayed_states = Replays[:,:,0]            
        replayed_actions = Replays[:,:,1]  
        # we select only experiences for a specific action
        replayed_states *= (replayed_actions == action)
        
        # array with all state indices
        expanded_states = np.arange(num_states)[na,na,:]
        expanded_replayed_states = replayed_states[:,:,na]


        # count over all replays within an episode the occurrence of each state (shape 500 x num_states)
        replayed_count += np.mean(np.sum(expanded_replayed_states == expanded_states, axis = 1), axis = 0)
    replayed_count /= (params['runs']) 
    return replayed_count

def visualize_CI_density(q, q_optimal, handles, beta, params):
    """ 
    This function plots average CI for each update.
    """
    fig, (axs0, axs1) = handles
    env = params['env']

    #fig, axs = handles

    mask = ~np.isnan(q_optimal)

    num_trials = q.shape[0]
    num_updates = q.shape[1] - 1
    num_states = q.shape[2]
    num_actions = q.shape[3]

    # shape is (trials, episodes, states, actions)
    error = q - q_optimal
    
    # plot CI per episode
    mse = np.mean(np.square(error), axis = (2,3))
    ci = np.zeros((num_trials, num_updates))
    for i in range(num_updates):
       ci[:,i] = np.maximum(mse[:,i+1] - mse[:,i], 0)
       if np.any(np.isnan(mse[:, i])):  # If NaN encountered
           print("np.isnan(mse[:, i]), i = ", i, num_updates)
       if np.any(np.isnan(mse[:, i+1])):  # If NaN encountered
           print("np.isnan(mse[:, i+1]), i+1 = ", i,num_updates)
    # average over all experiments
    ci_mean_episode = np.mean(ci, axis = 0)
    print("ci_mean_episode")

    scatter = axs0.scatter(np.arange(num_updates),ci_mean_episode, color=colors[beta], alpha=0.5, s=20, label = 'beta=%s'% (beta))
    axs0.axhline(y=0, color='black', linewidth=1)
    axs0.set_xticks(np.arange(0, num_updates + 1, 100))
    axs0.xaxis.grid(True, which='both')  # Enable grid for x-axis
    axs0.yaxis.grid(True, which='both')  # Enable grid for y-axis
    axs0.set_ylabel('Catastrophic Interference', fontsize=fontsize)
    axs0.set_xlabel('trials', fontsize=fontsize)
    axs0.tick_params(axis='both', which='major', labelsize=labelsize)


    # Density plot 
    sns.kdeplot(y = ci_mean_episode, ax=axs1, color = colors[beta], fill=True, common_norm=False)
 
    axs1.set_ylim(axs0.get_ylim())
    axs1.set_yticks([])  # Remove y-axis ticks from the density plot
    axs1.set_xticks([])
    axs1.set_xlabel('')
    png_name = 'CI_density_episodes_%s_as%s_oh%s_r%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0], params['runs'])


    if params['savefigs']:
        setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
        path = 'plots/' + setup
        if not os.path.exists(path):
            os.makedirs(path)
        beta = 0
        png_name = path + png_name
        print("saving fig as: ", png_name)
        fig.savefig(png_name, dpi=300,bbox_inches='tight')




def visualize_CI_density_old(q, q_optimal, handles, beta, params):
    """ 
    This function plots average CI for each update.
    """
    fig, (axs0, axs1) = handles
    env = params['env']

    #fig, axs = handles

    mask = ~np.isnan(q_optimal)

    num_trials = q.shape[0]
    num_updates = q.shape[1] - 1
    num_states = q.shape[2]
    num_actions = q.shape[3]

    # shape is (trials, episodes, states, actions)
    error = q - q_optimal
    
    # plot CI per episode
    mse = np.mean(np.square(error), axis = (2,3))
    ci = np.zeros((num_trials, num_updates))
    for i in range(num_updates):
       ci[:,i] = np.maximum(mse[:,i+1] - mse[:,i], 0)
       if np.any(np.isnan(mse[:, i])):  # If NaN encountered
           print("np.isnan(mse[:, i]), i = ", i, num_updates)
       if np.any(np.isnan(mse[:, i+1])):  # If NaN encountered
           print("np.isnan(mse[:, i+1]), i+1 = ", i,num_updates)
    # average over all experiments
    ci_mean_episode = np.mean(ci, axis = 0)
    print("ci_mean_episode")

    # each dot in scatter plot represents average CI in an episode
    #axs.scatter(np.arange(num_updates),ci_mean_episode, s= 3, color = colors[beta], label = 'beta=%s'% (beta))
    #axs.scatter(np.arange(num_updates),ci_mean_episode, s=70, alpha=0.03, color = colors[beta], label = 'beta=%s'% (beta))
    #axs.scatter(np.arange(num_updates),ci_mean_episode, s=70, facecolors='none', edgecolors=colors[beta], label = 'beta=%s'% (beta))
    #axs.scatter(np.arange(num_updates),ci_mean_episode, s= 10,  color = colors[beta], label = 'beta=%s'% (beta))
    scatter = axs0.scatter(np.arange(num_updates),ci_mean_episode, color=colors[beta], alpha=0.5, edgecolor='k', s=20, label = 'beta=%s'% (beta))

    #axs.plot(np.arange(num_updates),ci_mean_episode,color  = colors[beta])
    axs0.axhline(y=0, color='black', linewidth=1)
    axs0.set_xticks(np.arange(0, num_updates + 1, 100))
    axs0.xaxis.grid(True, which='both')  # Enable grid for x-axis
    axs0.yaxis.grid(True, which='both')  # Enable grid for y-axis
    axs0.set_ylabel('Catastrophic Interference', fontsize=fontsize)
    axs0.set_xlabel('trials', fontsize=fontsize)
    axs0.tick_params(axis='both', which='major', labelsize=labelsize)

    # Density plot 
    sns.kdeplot(y = ci_mean_episode, ax=axs1, color = colors[beta], fill=True, common_norm=False)


    png_name = 'CI_episodes_%s_as%s_oh%s_r%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0], params['runs'])
    plt.tight_layout()
    #plt.tight_layout()


    if params['savefigs']:
        setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
        path = 'plots/' + setup
        if not os.path.exists(path):
            os.makedirs(path)
        beta = 0
        png_name = path + png_name
        print("saving fig as: ", png_name)
        fig.savefig(png_name, dpi=300,bbox_inches='tight')


def visualize_CI(q, q_optimal, handles, beta, params):
    """ 
    This function plots average CI for each update.
    """
    
    env = params['env']

    fig, axs = handles

    mask = ~np.isnan(q_optimal)

    num_trials = q.shape[0]
    num_updates = q.shape[1] - 1
    num_states = q.shape[2]
    num_actions = q.shape[3]

    # shape is (trials, episodes, states, actions)
    error = q - q_optimal
    
    # plot CI per episode
    mse = np.mean(np.square(error), axis = (2,3))
    ci = np.zeros((num_trials, num_updates))
    for i in range(num_updates):
       ci[:,i] = np.maximum(mse[:,i+1] - mse[:,i], 0)
       if np.any(np.isnan(mse[:, i])):  # If NaN encountered
           print("np.isnan(mse[:, i]), i = ", i, num_updates)
       if np.any(np.isnan(mse[:, i+1])):  # If NaN encountered
           print("np.isnan(mse[:, i+1]), i+1 = ", i,num_updates)
    # average over all experiments
    ci_mean_episode = np.mean(ci, axis = 0)
    print("ci_mean_episode")

    # each dot in scatter plot represents average CI in an episode
    #axs.scatter(np.arange(num_updates),ci_mean_episode, s= 3, color = colors[beta], label = 'beta=%s'% (beta))
    #axs.scatter(np.arange(num_updates),ci_mean_episode, s=70, alpha=0.03, color = colors[beta], label = 'beta=%s'% (beta))
    #axs.scatter(np.arange(num_updates),ci_mean_episode, s=70, facecolors='none', edgecolors=colors[beta], label = 'beta=%s'% (beta))
    axs.scatter(np.arange(num_updates),ci_mean_episode, s= 10,  color = colors[beta], label = 'beta=%s'% (beta))

    #axs.plot(np.arange(num_updates),ci_mean_episode,color  = colors[beta])
    axs.axhline(y=0, color='black', linewidth=1)
    axs.set_xticks(np.arange(0, num_updates + 1, 100))
    axs.xaxis.grid(True, which='both')  # Enable grid for x-axis
    axs.yaxis.grid(True, which='both')  # Enable grid for y-axis
    axs.set_ylabel('Catastrophic Interference', fontsize=fontsize)
    axs.set_xlabel('trials', fontsize=fontsize)
    axs.tick_params(axis='both', which='major', labelsize=labelsize)
    
    png_name = 'CI_episodes_%s_as%s_oh%s_r%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0], params['runs'])
    plt.tight_layout()
    #plt.tight_layout()


    if params['savefigs']:
        setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
        path = 'plots/' + setup
        if not os.path.exists(path):
            os.makedirs(path)
        beta = 0
        png_name = path + png_name
        print("saving fig as: ", png_name)
        fig.savefig(png_name, dpi=300,bbox_inches='tight')

def load_R(beta,data_folder, num_states, params,action, episodes = (0,500)):
    """ 
    This function returns the replay frequency for each episode for each state for the specified action
    """
    num_runs = params['runs']
    num_replay = params['num_replay']
    Replay_matrix = np.empty((500*num_replay[0],32,4),dtype='float64')
    data_file = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
    replayed_count = np.zeros((params['runs'],500,num_states), dtype='float64')

    for run in range(params['runs']):
        data_path_replay = data_file + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, run)
        assert os.path.exists(data_path_replay), f"Data path '{data_path_replay}' does not exist"
        # load replay_batches
        with open(data_path_replay, 'rb') as handle:
            print("Loading... Replay_matrix.shape: ", Replay_matrix.shape)
            # 25000 x 32 x 4
            Replay_matrix[:] = pickle.load(handle)#replay_batches
        Replays = Replay_matrix.reshape(500, num_replay[0]*32, -1)

        # shape 500 x 1600
        replayed_states = Replays[:,:,0]            
        replayed_actions = Replays[:,:,1]  
        # we select only experiences for a specific action
        replayed_states *= (replayed_actions == action)
        expanded_states = np.arange(num_states)[na,na,:]

        # array with all state indices
        expanded_states = np.arange(num_states)[na,na,:]
        expanded_replayed_states = replayed_states[:,:,na]

        # count over all replays within an episode the occurrence of each state (shape num_runs x 500 x num_states)
        replayed_count[run] = np.sum(expanded_replayed_states == expanded_states, axis = 1)
    return replayed_count

def load_R_all(beta,data_folder, num_states, params,action, episodes = (0,500)):
    """ 
    This function returns the R_matrix for all actions and the actions for masking
    """
    num_runs = params['runs']
    num_replay = params['num_replay']
    Replay_matrix = np.empty((500*num_replay[0],32,4),dtype='float64')
    data_file = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
    replayed_count = np.zeros((params['runs'],500,num_states), dtype='float64')
    count_a = 0
    count_not_a = 0
    for run in range(params['runs']):
        data_path_replay = data_file + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, run)
        assert os.path.exists(data_path_replay), f"Data path '{data_path_replay}' does not exist"
        # load replay_batches
        with open(data_path_replay, 'rb') as handle:
            print("Loading... Replay_matrix.shape: ", Replay_matrix.shape)
            # 25000 x 32 x 4
            Replay_matrix[:] = pickle.load(handle)#replay_batches
        Replays = Replay_matrix.reshape(500, num_replay[0]*32, -1)

        # shape 500 x 1600
        replayed_states = Replays[:,:,0]            
        replayed_actions = Replays[:,:,1]  
        # we select only experiences for a specific action
        replayed_states *= (replayed_actions == action)
        expanded_states = np.arange(num_states)[na,na,:]

        # array with all state indices
        expanded_states = np.arange(num_states)[na,na,:]
        expanded_replayed_states = replayed_states[:,:,na]

        # count over all replays within an episode the occurrence of each state (shape num_runs x 500 x num_states)
        replayed_count[run] = np.sum(expanded_replayed_states == expanded_states, axis = 1)
    return replayed_count


def load_Q(beta, data_folder, num_states,num_actions, params):
    """
    This function loads the Q functions from multiple runs
    """
    Q_matrix = np.empty((params['runs'], 501,num_states,num_actions), dtype='float64')
    running_env = params['env']
    r_type = params['replay_type']
    data_file = data_folder + '/beta_%s/%s+%s' % (beta, params['num_replay'][0], params['num_replay'][1])
    for run in range(params['runs']):
        data_path = data_file + '/Q_values_%s_%s_%s.pickle' % (r_type, running_env, run)
        assert os.path.exists(data_path), f"Data path '{data_path}' does not exist for run {run}"
        # load q functions
        with open(data_path, 'rb') as handle:
            Q_matrix[run] = pickle.load(handle)
    return Q_matrix

def load_T(beta, data_folder, num_states,num_actions, params):
    """
    This function loads the Replay Trajectories from multiple runs
    """
    T_matrix = np.empty((params['runs'], 500),dtype='float32')
    running_env = params['env']
    r_type = params['replay_type']
    data_file = data_folder + '/beta_%s/%s+%s' % (beta, params['num_replay'][0], params['num_replay'][1])
    for run in range(params['runs']):
        data_path = data_file + '/TrainingTrajs_%s_%s_%s.pickle' % (r_type, running_env, run)
        assert os.path.exists(data_path), f"Path doesn't exist: {data_path}"

        num_steps = []

        with open(data_path, 'rb') as handle:

            for item in pickle.load(handle):
                num_steps.append(len(item))

        T_matrix[run] = num_steps
    return T_matrix




if __name__ == "__main__":

    # number of trials over which we average
    params = {'replay_colors': ['black', 'orange'], 'linestyles': ['solid', 'solid'],
              'replay_type': 'SR_AU', 'runs': 1, 'savefigs' : True, 'env' : 'TunnelMaze_LV1',
              'action_space' : 3, 'one_hot' : True, 'num_replay': [50,0], 'extended_by':0}

    data_folder = 'data/sequential_replay_1_torch/action_space_%s_one_hot_%s' % (params['action_space'], params['one_hot'])
    if params['extended_by']:
        data_folder = 'data/sequential_replay_1_torch_%s/action_space_%s_one_hot_%s' % (params['extended_by'], params['action_space'], params['one_hot'])

    if len(sys.argv) > 1:
        data_folder = sys.argv[1] + data_folder

    betas = [1]#1,2,5,10]#0
    
    
    num_replay = [params['num_replay'], 0]

    r_type = 'SR_AU'

    running_env = params['env']

    data_path_optimal_q = data_folder + '/%s_optimal_Q.pickle' % (running_env) 

    # Whether the plots should be stored as png

    # load and process optimal q-function
    with open(data_path_optimal_q, 'rb') as handle:
        q_optimal = pickle.load(handle)

        # create mask in case there are invalid states   
        mask = ~np.isnan(q_optimal)
    num_states = q_optimal.shape[0]
    num_actions = q_optimal.shape[1]

    # escapte latency
    draw_el = False
    # Variance from Q*, CI per episode, CI per state per action
    draw_CI = False
    # CI per state per action 
    draw_CIsa = False #and draw_CI
    draw_replay_frequency = True
    # AC over averaged Variance
    draw_AC_replay_variance = False
    draw_AC_nicolas = False
    draw_AC_single = False
    draw_AC = False
    # AC averaged over all transitions and experiments
    draw_AC_avg = False
    relative_for = -1 # if this is a valid beta, replay frequencies are plotted relative to this beta
    # otherwise plot absolute replay frequency

    actions = range(num_actions)
    #action = 1
    
    beta = 0

    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.tsa.stattools import acf

    # Start tracing Python memory allocations
    #tracemalloc.start()
    test_delete = False
    
    
    
    time_lags = 500
    
    states = 540
    action = 0
    if False:
        R_file_beta = 'R_%s' % beta
        for beta in [10]:
            #fig, axs = plt.subplots()
            #if not os.path.exists(R_file_beta):
            """
            def f(x, y):
                return np.sin(np.sqrt(x ** 2 + y ** 2))

            x = np.linspace(-6, 6, 30)
            y = np.linspace(-6, 6, 30)

            X, Y = np.meshgrid(x, y)
            Z = f(X, Y)"""
            
            #print(X.shape, Y.shape, Z.shape)
            #A = np.array([X,Y,Z])
            # 3D visualization



            Y = np.arange(states)
            X = np.arange(time_lags)
            X,Y = np.meshgrid(X,Y)
            
            
            Z = np.random.randint(2,5,(states,time_lags))
            
            """   
            
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlabel('time_lag')
            ax.set_ylabel('state')
            ax.set_zlabel('AC')
            
            """
            Z = np.empty((states,time_lags))
            all_AC = np.empty((params['runs'],time_lags))
            R_matrix = load_R(beta,data_folder,num_states,params, action=action)
            print("R_matrix.shape", R_matrix.shape)
            for state in range(states):
                for i in range(params['runs']):
                    # shape runs x episodes x states
                    R = R_matrix[i,:,state]
                    # Adding very small noise to avoid zero variance in acf calculation
                    R += np.random.normal(0, 1e-6, size=R.shape)
                    print("R.shape ",R.shape)

                    print("Sum of R:", np.sum(R))
                    print("Mean of R:", np.mean(R))
                    print("Variance of R:", np.var(R))
                    ac = acf(R, nlags = time_lags-1)
                    print("ac.shape = ",ac.shape)
                    all_AC[i] = ac
                    

                    #axs.stem(ac)
                    #axs.set_title('AC by hand beta %s' % beta)
                ac_avg = np.mean(all_AC, axis = 0)
                Z[state] = ac_avg
                #print("ac = ", ac.shape)
            
            #plt.figure(figsize=(10, 8))
            
            #print("Z.shape = ", Z.shape, Z.dtype)
            fig,ax = plt.subplots()
            states = 540
            time_lags = 500
            Z = Z[:states,:time_lags]#np.random.rand(states,time_lags)
            print("z type = ", Z.dtype)
            
            #sns.heatmap(Z, ax = ax, cmap='viridis', annot=False, fmt=".2f", cbar=True)
            #sns.heatmap(Z, ax=ax, cmap='coolwarm', annot=False, fmt=".2f", cbar=True, center=0)
            #sns.heatmap(Z, ax=ax, cmap="PiYG", annot=False,center = 0)
            
            
            #current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
            print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
            #raise ValueError
            plt.xlabel('Time Lag')            
            plt.ylabel('State')
            plt.title('Heatmap of State vs Time Lag')
            ax.set_title('AC: beta %s' % beta)

            #ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
            #    cmap='viridis', edgecolor='none')
            plt.savefig('Heat_AC_b%s_r%s_s%s_a%s.png' % (beta,params['runs'],states,action))
            #plt.savefig('AC_%s_%s_%s_%s.png' % (beta,params['runs'],state,action)) 
            
    if draw_el:

        handles = plt.subplots(figsize=(6.4,4.8))
        
        plot_name = 'LC_EL_%s_as%s_oh%s_r%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0],params['runs'])

        for beta in betas:
            data_file = data_folder + '/beta_%s/%s+%s' % (beta, params['num_replay'][0], params['num_replay'][1])
            
            # load Trajectories
            T_matrix = load_T(beta,data_folder,num_states,num_actions,params)


            visualize_LC(T_matrix, handles, beta, running_env, params['action_space'], num_replay, x_label = 'trials', y_label = '# of time steps', plot_name = plot_name, show_legend= False)
    # Variance from Q* and CI per episode
    if draw_CI: 

        #handles_CI_episodes = plt.subplots(figsize=(6.4,4.8))        
        #handles_LC_std = plt.subplots(figsize=(6.4,4.8))
        #handles_LC_all = plt.subplots(figsize=(6.4,4.8))
        handles_CI_episodes = plt.subplots(1, 2, figsize=(8.4, 4.8), gridspec_kw={'width_ratios': [3, 1], 'wspace':0.05})

        for beta in betas:

            # Load Q
            Q_matrix = load_Q(beta,data_folder,num_states,num_actions,params)

            error = Q_matrix - q_optimal

            mse = np.mean(np.square(error), axis = (2,3))
            
            #visualize_CI(Q_matrix, q_optimal, handles_CI_episodes, beta, params)

            visualize_CI_density(Q_matrix, q_optimal, handles_CI_episodes, beta, params)
            #visualize_LC(mse, handles_LC_std, beta, running_env, params['action_space'], num_replay, x_label='trials',y_label='MSE', show_std = True)
            #visualize_LC(mse, handles_LC_all, beta, running_env, params['action_space'], num_replay, x_label='trials',y_label='MSE', show_all = True)


    draw_absolute_replay_frequency = False
    if draw_absolute_replay_frequency:
        # AC w.r.t. average variance in replay 
        betas = [1,2,5,10]
        actions = list(range(num_actions))
        print("actions = ", actions)
        r_type = params['replay_type']
        running_env = params['env']
        
        num_runs = params['runs']
        num_replay = params['num_replay']
        Replay_matrix = np.empty((500*num_replay[0],32,4),dtype='float64')
        for beta in betas:
            data_file = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
            replayed_count = np.zeros((500,num_states,len(actions)), dtype='float64')
            Replay_variance = np.zeros((500))
            for run in range(params['runs']):
                data_path_replay = data_file + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, run)
                
                assert os.path.exists(data_path_replay), f"Data path '{data_path_replay}' does not exist"
                # load replay_batches
                with open(data_path_replay, 'rb') as handle:
                    print("Loading... Replay_matrix.shape: ", Replay_matrix.shape)
                    # 25000 x 32 x 4
                    Replay_matrix[:] = pickle.load(handle)#replay_batches
                
                Replays = Replay_matrix.reshape(500, num_replay[0]*32, -1)

                # shape 500 x 1600
                replayed_states = Replays[:,:,0]            
                replayed_actions = Replays[:,:,1]  

                expanded_states = np.arange(num_states)[na,na,:]
                for action in actions:
                    # we select only experiences for a specific action
                    replayed_states *= (replayed_actions == action)

                    # array with all state indices
                    expanded_replayed_states = replayed_states[:,:,na]

                    # count over all replays within an episode the occurrence of each state (shape num_runs x 500 x num_states)
                    replayed_count[:,:,action] = np.sum(expanded_replayed_states == expanded_states, axis = 1)

            #replayed_count[]
            
            #print(R_variance[beta].shape)
            fig,axs = plt.subplots()
            plot_acf(Replay_variance, ax = axs, lags = 500-1,title='Autocorrelation beta %s' % beta)
            setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
            path = 'plots/' + setup
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + 'actual_AC_var_b%s_r%s_all_transitions.png' % (beta,params['runs']))
            

    # plot how often each state action was replayed on average
    if draw_replay_frequency:
        # compare two betas
        R_counts = {beta: np.zeros((num_states,num_actions)) for beta in betas}
        num_replay = params['num_replay']
        Replay_matrix = np.empty((500*num_replay[0],32,4),dtype='float64')
        vmin, vmax =  None, None
        for beta in betas:
            data_file = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
            replayed_count = np.zeros((500,num_states,len(actions)), dtype='float64')
            Replay_variance = np.zeros((500))
            for run in range(params['runs']):
                data_path_replay = data_file + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, run)
                
                assert os.path.exists(data_path_replay), f"Data path '{data_path_replay}' does not exist"
                # load replay_batches
                with open(data_path_replay, 'rb') as handle:
                    print("Loading... Replay_matrix.shape: ", Replay_matrix.shape)
                    # 25000 x 32 x 4
                    Replay_matrix[:] = pickle.load(handle)#replay_batches
                Replays = Replay_matrix.reshape(500, num_replay[0]*32, -1)
                # shape 500 x 1600
                replayed_states = Replays[:,:,0]            
                replayed_actions = Replays[:,:,1]  

                expanded_states = np.arange(num_states)[na,na,:]
                # This section includes a bug (reported in the appendix)!
                for action in actions:
                    # we select only experiences for a specific action
                    replayed_states *= (replayed_actions == action)
                    #print("replayed_states = ", replayed_states[:10,:10])

                    # array with all state indices
                    expanded_replayed_states = replayed_states[:,:,na]

                    # count over all replays within an episode the occurrence of each state (shape num_runs x 500 x num_states)

                    replayed_count[:,:,action] = np.sum(expanded_replayed_states == expanded_states, axis = (1))
                    #print("replayed_count = ", replayed_count[:10,:10,action])

                #raise ValueError
            # average replay frequency
            if np.any(np.isnan(replayed_count)):
                print("np.any(np.isnan(replayed_count)) shape = ", replayed_count.shape)
                print("replayed_count = ", np.where(np.isnan(replayed_count)))
            R_counts[beta] += np.sum(replayed_count,axis = 0)/params['runs']
            
            vmin = np.min(R_counts[beta]) if vmin == None else np.minimum(vmin, np.min(R_counts[beta]))
            vmax = np.maximum(np.percentile(R_counts[beta],90),1) if vmax == None else np.maximum(vmax, np.percentile(R_counts[beta],90))
            #R_counts[beta] /= params['runs']
        #vmin /= params['runs']
        #vmax /= params['runs']
            for action in actions:

                fig, axs = plt.subplots()
                norm = Normalize(vmin = vmin, vmax = vmax, clip = False)    
                cmap = plt.get_cmap('turbo') 
                plot_states(fig,axs,cmap,norm,R_counts[beta][:,action],params,vmin,vmax)

                if params['savefigs']:
                        png_name = 'R__absolute_%s_as%s_oh%s_r%s_b%s_a%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0],beta, action,params['runs'])
                        setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
                        path = 'plots/' + setup
                        if not os.path.exists(path):
                            os.makedirs(path)
                        png_name = path + png_name
                        print("saving fig as: ", png_name)
                        fig.savefig(png_name, dpi=300,bbox_inches='tight')

    # CI for all states and actions either absolute or relative 
    if draw_CIsa:
        # compare two betas
        
        relative_for = -1
        ci_sa = {beta: np.empty((num_actions, num_states)) for beta in betas}
        print("draw_CIsa")
        vmin = None
        vmax = None
        actions = [0,1]
        betas = [1,10]
        params['runs'] = 2
        for action in actions:
            for beta in betas:

                # Load replay frequency for each state
                Q_matrix = load_Q(beta, data_folder, num_states, num_actions, params)
                ci_matrix = get_CI_sa(Q_matrix, q_optimal, action)
                ci = np.mean(np.sum(ci_matrix, axis = 1), axis = (0))
                ci_sa[beta][action] = ci
                vmin = np.min(ci) if vmin == None else np.minimum(vmin, np.min(ci))
                vmax = np.percentile(ci,90) if vmax == None else np.maximum(vmax, np.percentile(ci,90))
            vmin = np.around(vmin, decimals=1)
            vmax = np.around(vmax, decimals=1)
        for action in actions:

            for beta in betas:
                # show replay frequency relative to specified beta
                if relative_for in betas:
                    if beta == relative_for:
                        continue
                    fig, axs = plt.subplots()

                    # num_runs x num_episodes x num_states
                    ci_relative = np.where(ci_sa[beta][action] == 0, np.where(ci_sa[relative_for][action] == 0, 1, np.inf), ci_sa[relative_for][action] / ci_sa[beta][action])
                    
                    count_g = np.sum(ci_relative > 1) 
                    count_l = np.sum(ci_relative < 1)
                    count_e = np.sum(ci_relative == 1)
                    print("shape = " ,ci_relative.shape, ci_relative)
                    print("greater: ", count_g,"less: ", count_l,"equal: ", count_e)
                    print("minimum: ", np.min(ci_relative))

                    cmap = plt.get_cmap('bwr')  
                    norm = Normalize(vmin = 0, vmax = 2, clip = False)
                    plot_states(fig,axs,cmap,norm,ci_relative,params)

                    if params['savefigs']:
                        png_name = 'CI_states_relativefor%s_%s_as%s_oh%s_r%s_b%s_a%s_avg%s.png'  % (relative_for,params['env'], params['action_space'], params['one_hot'], params['num_replay'][0], beta, action, params['runs'])
                        setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
                        path = 'plots/' + setup
                        if not os.path.exists(path):
                            os.makedirs(path)
                        png_name = path + png_name
                        print("saving fig as: ", png_name)
                        fig.savefig(png_name, dpi=300,bbox_inches='tight')

                else:
                    fig,axs = plt.subplots()

                    norm = Normalize(vmin = vmin, vmax = vmax, clip = False)
                        
                    
                    cmap = plt.get_cmap('rainbow')  

                    plot_states(fig,axs,cmap,norm,ci_sa[beta][action],params,vmin,vmax)
                    
                    print(ci_sa[beta])
                    if params['savefigs']:
                        png_name = 'CI_states_%s_as%s_oh%s_r%s_b%s_a%s_avg%s.png'  % (params['env'], params['action_space'], params['one_hot'], params['num_replay'][0], beta, action, params['runs'])
                        setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
                        path = 'plots/' + setup
                        if not os.path.exists(path):
                            os.makedirs(path)
                        png_name = path + png_name
                        print("saving fig as: ", png_name)
                        fig.savefig(png_name, dpi=300,bbox_inches='tight')


    if draw_AC_replay_variance:
        # AC w.r.t. average variance in replay 
        
        betas = [1,2,5,10]
        actions = list(range(num_actions))
        print("actions = ", actions)
        r_type = params['replay_type']
        running_env = params['env']
        
        num_runs = params['runs']
        num_replay = params['num_replay']
        Replay_matrix = np.empty((500*num_replay[0],32,4),dtype='float64')
        for beta in betas:
            data_file = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
            replayed_count = np.zeros((500,num_states,len(actions)), dtype='float64')
            Replay_variance = np.zeros((500))
            for run in range(params['runs']):
                data_path_replay = data_file + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, run)
                
                assert os.path.exists(data_path_replay), f"Data path '{data_path_replay}' does not exist"
                # load replay_batches
                with open(data_path_replay, 'rb') as handle:
                    print("Loading... Replay_matrix.shape: ", Replay_matrix.shape)
                    # 25000 x 32 x 4
                    Replay_matrix[:] = pickle.load(handle)#replay_batches
                
                Replays = Replay_matrix.reshape(500, num_replay[0]*32, -1)

                # shape 500 x 1600
                replayed_states = Replays[:,:,0]            
                replayed_actions = Replays[:,:,1]  

                expanded_states = np.arange(num_states)[na,na,:]
                for action in actions:
                    # we select only experiences for a specific action
                    replayed_states *= (replayed_actions == action)

                    # array with all state indices
                    expanded_replayed_states = replayed_states[:,:,na]

                    # count over all replays within an episode the occurrence of each state (shape num_runs x 500 x num_states)
                    replayed_count[:,:,action] = np.sum(expanded_replayed_states == expanded_states, axis = 1)

                # add up variance over all transitions
                print("replayed_count.shape",replayed_count.shape)
                print("var shpae = ",np.var(replayed_count, axis = (1,2)).shape)
                Replay_variance += np.var(replayed_count, axis = (1,2))
            # average replay_varianc
            Replay_variance /= params['runs']
            print("Replay_variance.shape = ", Replay_variance.shape)
            
            #print(R_variance[beta].shape)
            fig,axs = plt.subplots()
            plot_acf(Replay_variance, ax = axs, lags = 500-1,title='Autocorrelation beta %s' % beta)
            setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
            path = 'plots/' + setup
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + 'actual_AC_var_b%s_r%s_all_transitions.png' % (beta,params['runs']))
    
    if draw_AC:
        def ac_total(data):
            dists = []
            mean = np.mean(data)
            print("mean = ", mean)
            for i in range(-(t-1), t):
                #shifted = np.roll(data, i, axis=0)
                # zero wrapped around values
                if i > 0:
                    #shifted[(500 + i):] = 0.
                    x = (data[i:]-mean) * (data[:-i]-mean)
                elif i < 0:
                    #shifted[:i] = 0.
                    x = (data[:i]-mean) * (data[-i:]-mean)
                else:
                    x = (data-mean)**2
                # correlation
                x /= np.var(data) * data.size
                dists.append(np.sum(x))

            return np.array(dists)
        def ac_single(data):
            dists = []
            mean = np.mean(data)
            for i in range(-499, 499):
                if i > 0:
                    #shifted[(500 + i):] = 0.
                    x = (data[i:]-mean) * (data[:-i]-mean)
                elif i < 0:
                    #shifted[:i] = 0.
                    x = (data[:i]-mean) * (data[-i:]-mean)
                else:
                    x = (data-mean)**2
                # covariance
                #x = (data - mean) * (shifted - mean)
                # correlation
                x /= np.var(data)# * x.size)#data.size
                print("x = ",len(x), x.size, x.shape)
                dists.append(np.mean(x))
            return np.array(dists)
        load_R = True
        betas = [1,10]
        params['runs'] = 2
        actions = list(range(num_actions))
        r_type = params['replay_type']
        running_env = params['env']
        
        num_runs = params['runs']
        num_replay = params['num_replay']
        Replay_matrix = np.empty((500*num_replay[0],32,4),dtype='float64')

        if not draw_AC_single:
            acf = np.empty((params['runs'],999))

        def ac_total(data):
            mean = np.mean(data)
            print("mean = ", mean, data.size, data.shape)
            dists = []
            for i in range(-499,500):
                
                if i > 0:
                    x = (data[i:]-mean) * (data[:-i]-mean)
                elif i < 0:
                    x = (data[:i]-mean) * (data[-i:]-mean)
                else:
                    x = (data-mean)**2
                x /= np.var(data) * data.size
                dists.append(np.sum(x))
            return np.array(dists)
        for beta in betas:
            data_mean = np.zeros((500,num_states*len(actions)), dtype='float64')
            if load_R:
                
                print(f"Beta = {beta}")
                data_file = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
                replayed_count = np.zeros((500,num_states,len(actions)), dtype='float64')
                Replay_variance = np.zeros((500))
                for run in range(params['runs']):
                    data_path_replay = data_file + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, run)
                    
                    assert os.path.exists(data_path_replay), f"Data path '{data_path_replay}' does not exist"
                    # load replay_batches
                    with open(data_path_replay, 'rb') as handle:
                        print("Loading... Replay_matrix.shape: ", Replay_matrix.shape)
                        # 25000 x 32 x 4
                        Replay_matrix[:] = pickle.load(handle)#replay_batches
                    
                    Replays = Replay_matrix.reshape(500, num_replay[0]*32, -1)
                    # shape 500 x 1600
                    replayed_states = Replays[:,:,0]            
                    replayed_actions = Replays[:,:,1]  
                    expanded_states = np.arange(num_states)[na,na,:]
                    print("replayed_actions= ",replayed_actions[:10,:10])
                    for action in actions:
                        # we select only experiences for a specific action
                        replayed_states *= (replayed_actions == action)
                        print("relayed_states= ", replayed_states[:10,:10])
                        # array with all state indices
                        expanded_replayed_states = replayed_states[:,:,na]

                        # count over all replays within an episode the occurrence of each state (500 x num_states x num_actions)
                        replayed_count[:,:,action] = np.sum(expanded_replayed_states == expanded_states, axis = 1)
                    print(data_mean.shape, replayed_count.shape)
                    data_mean += replayed_count.reshape(500,num_states*num_actions)

                    draw_AC_single = False
                    if draw_AC_single:
                    
                        # Create the subplot
                        fig, axs = plt.subplots(2)

                        axs[0].set_title(f'Beta = {beta}')

                        ac = np.empty((num_states * num_actions, 500))
                        ac_by_hand = np.empty((num_states * num_actions, 500))
                        for i in range(num_states * num_actions):
                            #print("shape = ", replayed_count[:,i].shape)
                            ac[i] = acf(replayed_count[:,i],nlags = 500)
                            print("shape replayed count = ", replayed_count.shape)
                            m = np.mean(replayed_count[:,i])
                            print("var = ", np.sum((replayed_count[:,i]-m)**2))
                            #print("mean = ", m)
                            for j in range(499):
                                #if np.sum((replayed_count[j:,i]-m) * (replayed_count[:500-j,i]-m)) < 0:
                                #    
                                #    #print(" < 0 : ", np.sum((replayed_count[j:,i]-m) * (replayed_count[:500-j,i]-m)))
                                ac_by_hand[i,j] = np.sum((replayed_count[j:,i]-m) * (replayed_count[:500-j,i]-m))
                                ac_by_hand[i,j] /= np.sum((replayed_count[:,i]-m)**2)
                            #print(ac[i], np.amax(ac[i]))
                        #v_min = np.amin(ac)
                        #print(np.amax(ac[:,0]))
                        #v_max = np.amax(ac)
                            print("np.min(ac_by_hand), np.max(ac_by_hand)", np.min(ac_by_hand[i,:]), np.max(ac_by_hand[i,:]))

                        c = axs[0].pcolor(ac, cmap='bwr', vmin=-1, vmax=1)
                        c_by_hand = axs[1].pcolor(ac_by_hand, cmap='bwr', vmin=-1, vmax=1)

                        fig.colorbar(c, ax=axs[0])
                        fig.colorbar(c_by_hand, ax=axs[1])
                        axs[0].set_xlabel('Time')
                        axs[0].set_ylabel('Signal')
                        axs[1].set_xlabel('Time')
                        axs[1].set_ylabel('Signal')
                        file = 'Ac_single_b%s_r%s_all_transitions.png' % (beta,params['runs'])
                    # draw one AC curve for all transitions
                    else:
                        # get overall mean

                        
                        #for i in range(params['runs']):
                        #print("data = ", data[:10,:10])
                        #print("data.shape = ", data.shape)
                        acf[run] = ac_total(replayed_count)
                        #fig, axs = plt.subplots()
                        #axs.plot(acf)
                        #axs[0].plot(np.arange(998)-499, ac)
                                                
                        #v_min = np.amin(np.mean(data,axis =0))
                        #v_max = np.amax(np.mean(data,axis =0))

                if not draw_AC_single:
                    #plt.figure(1)
                    fig, axs = plt.subplots(2)
                    #plt.subplots_adjust(wspace=0.5)
                    #plt.subplot(1, 1, 1)
                    axs[0].set_title(f'Beta = {beta}')
                    data_mean /= params['runs']
                    vmin = 0#np.min(data_mean)
                    vmax = 100#np.max(data_mean)
                    print("vmax = ", vmax)
                    axs[0].pcolor(data_mean.T, cmap='hot', vmin=vmin, vmax=vmax)
                    #plt.xlabel('Time')
                    #plt.ylabel('Signal')
                    #plt.subplot(1, 2, 2)
                    #plt.title('Noise')
                    #plt.xlabel('Time')
                    #plt.ylabel('Signal')
                    #plt.show()
                    #plt.figure(2)
                    #plt.title('Auto-Correlation')
                    axs[1].axhline(0, color='grey', linestyle='--')
                    axs[1].axhline(1, color='grey', linestyle='--')
                    print(acf.shape)
                    axs[1].plot(np.arange(999) - 499, np.mean(acf,axis = 0), color='r', label='Bump')
                    #plt.plot(np.arange(202) - 101, ac(signal_noise), color='b', label='Noise')
                    #axs[1].gca().axes.spines['top'].set_visible(False) # type: ignore
                    #axs[1].gca().axes.spines['right'].set_visible(False) # type: ignore
                    #plt.xlabel('Time Offset')
                    #plt.ylabel('Correlation')
                    #plt.legend()
                    file = 'Ac_total_b%s_r%s_all_transitions.png' % (beta,params['runs'])


            replayed_count = replayed_count.reshape(-1,500,num_states*num_actions)


            

        params['savefigs'] = True
        if params['savefigs']:
            setup = '%s_%s_%s/' % (params['env'],params['action_space'],params['one_hot'])
            path = 'plots/' + setup
            if not os.path.exists(path):
                os.makedirs(path)
            png_name = path + file 
            print("saving fig as: ", png_name)
            plt.savefig(png_name)


    plt.show()
