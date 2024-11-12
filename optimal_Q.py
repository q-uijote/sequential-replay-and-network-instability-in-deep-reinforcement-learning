import numpy as np
from numpy import sin, cos, pi
import heapq

def transition(state, action, state_dict):
    """
    transition from state to new_state by taking action 
    """
    x, y, angle = state_dict['states'][state]

    if action == 0:
        # forward
        x_next, y_next = x + int(cos(angle/180 * pi)), y + int(sin(angle/180 * pi))
        angle_next = angle
    elif action == 1:
        # rotate left
        x_next, y_next = x, y
        angle_next = angle + 90
        if angle_next == 180:
            angle_next = -180
    elif action == 2:
        # rotate right
        x_next, y_next = x, y
        angle_next = angle - 90
        if angle_next == -270:
            angle_next = 90
    elif action == 3:
        # move left
        x_next, y_next = x + int(sin(-angle/180*pi)), y + int(cos(angle/180*pi))
        angle_next = angle
    elif action == 4:
        # move right
        x_next, y_next = x + int(sin(angle/180*pi)), y - int(cos(angle/180*pi))
        angle_next = angle
    elif action == 5:
        # move backward
        x_next, y_next = x - int(cos(angle/180*pi)), y + int(sin(-angle/180*pi))
        angle_next = angle
    state = (x_next, y_next, angle_next)

    if state in state_dict['states']:
        # check whether action is valid
        return state_dict['states'].index(state)
    else: 
        # not a valid transition
        return -1
    

def dijkstra(state_dict, first_goal_index, num_of_states,num_of_actions):
    
    assert num_of_actions in [3,4,6], "invalid action space"


    distances = {state : float('inf') for state in range(num_of_states)}
    q_values = {state : 0 for state in range(num_of_states)}

    pq = []
    
    for goal_state in range(first_goal_index ,first_goal_index + 4):
        # there are 4 goal states
        distances[goal_state] = 0
        q_values[goal_state] = 0
        pq.append((0, goal_state))

 
    # stores distances and state_indices only
    heapq.heapify(pq)

    while pq:
        current_distance, current_state = heapq.heappop(pq)
        
        if current_distance > distances[current_state]:
            continue
        
        for action in range(num_of_actions):
            if num_of_actions == 4 and action > 0: # for |action space| = 4
                action += 2 # map turn left -> move left, turn right -> turn right, move left -> backward
            next_state = transition(current_state, action, state_dict)
            if next_state >= 0:
                distance = current_distance + 1
                if distance < distances[next_state]:
                    distances[next_state] = distance
                    heapq.heappush(pq, (distance, next_state))

    # if action space = 3, only left turn, right turn, and forward movement are allowed
    # we therefore swap calculated distances for states with opposing angle (can be easily proven)
    if num_of_actions == 3:
        for state in range(0,num_of_states,4):
            distances[state], distances[state + 2] = distances[state + 2], distances[state] 
            distances[state + 1], distances[state + 3] = distances[state + 3], distances[state + 1] 
    
    return distances

def optimal_q_values(state_dict, first_goal_index, num_of_states, num_of_actions, gamma):

    distances = dijkstra(state_dict, first_goal_index, num_of_states, num_of_actions)

    # Initialize array of shape num_of_states x num_of_actions with None values and terminal states with 0
    optimal_q_values = np.full((num_of_states, num_of_actions), np.nan)

    # Calculate optimal Q values
    # since reward = 1 for the transition to the final state and 0 otherwise, the calculation reduces to
    # q(s,a) = 1 * gamma ** k 
    # where k is distance to the goal state
    for state in range(num_of_states):
        for action in range(num_of_actions):
            next_state = transition(state, action, state_dict)
            # update q_values only for valid tuples (s,a) 
            if next_state >= 0:
                optimal_q_values[state,action] = 1 * gamma ** distances[next_state]
            else:
                # if next_state doesn't exist, the return is 0
                optimal_q_values[state,action] = 1 * gamma ** distances[state]
    
    optimal_q_values[first_goal_index : first_goal_index + 4, :] = 0

    for i in range(num_of_states):
        print("idx = ", i // 4, " x,y,angle =  ", state_dict['states'][i], " distance =  ", distances[i], "q_values = ", optimal_q_values[i])
    return optimal_q_values

    

