'''
Main function of 2.1.
'''
#TODO: Add statistics of data
import state
import numpy as np
import sys



FLT_MAX = sys.float_info.max
# Possible actions
MOVE_UP = -0.04 # index 0
MOVE_DOWN = 0.04 # index 1
POSSIBLE_MOVE = [MOVE_UP, MOVE_DOWN]
#gamma = 0.9
#decay_constant = 60.0
#num_e = 5

# State-action array, where value is the Q.
# Q(s, a) -- ball_x, ball_y, vx, vy, paddle_y, gameover, action
Q_VALUE = np.zeros((12, 12, 2, 3, 12, 2, 2))
# N(s, a)
N_ACTION = np.zeros((12, 12, 2, 3, 12, 2, 2))

def get_state_Q(_state, action):
    '''
    Get the Q-value of the input state.
    '''
    dis_state = _state.discrete_state()
    bx = dis_state[0]
    by = dis_state[1]
    vx = dis_state[2]
    vy = dis_state[3]
    p_yr = dis_state[4]
    gameover = dis_state[5]
    return Q_VALUE[bx][by][vx][vy][p_yr][gameover][action]


def set_state_Q(_state, action, value):
    '''
    Set the Q-value of the input state
    '''
    dis_state = _state.discrete_state()
    bx = dis_state[0]
    by = dis_state[1]
    vx = dis_state[2]
    vy = dis_state[3]
    p_yr = dis_state[4]
    gameover = dis_state[5]
    Q_VALUE[bx][by][vx][vy][p_yr][gameover][action] = value


def get_state_N(_state, action):
    '''
    Get the N of the input state.
    '''
    dis_state = _state.discrete_state()
    bx = dis_state[0]
    by = dis_state[1]
    vx = dis_state[2]
    vy = dis_state[3]
    p_yr = dis_state[4]
    gameover = dis_state[5]
    #print(str(bx) + ' ' + str(by) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(p_yr) + ' ' + str(gameover))
    return N_ACTION[bx][by][vx][vy][p_yr][gameover][action]


def set_state_N(_state, action, value):
    '''
    Set the N of the input state.
    '''
    dis_state = _state.discrete_state()
    bx = dis_state[0]
    by = dis_state[1]
    vx = dis_state[2]
    vy = dis_state[3]
    p_yr = dis_state[4]
    gameover = dis_state[5]
    N_ACTION[bx][by][vx][vy][p_yr][gameover][action] = value

def learn(gamma, decay_constant, num_e):
    '''
    A single learning trail.
    '''
    init_state = state.State()
    curr_state = init_state
    while 1:
        # From current state s, select an action a.
        curr_state_discrete = curr_state.discrete_state()
        max_exp_func = -1
        over = curr_state_discrete[5]
        if over == 1:
            break
        action = 0
        new_num_act = 0
        for i in range(2):
            num_action = get_state_N(curr_state, i)
            if num_action < num_e:
                exp_func = FLT_MAX
            else:
                exp_func = get_state_Q(curr_state, i)
            if exp_func > max_exp_func:
                max_exp_func = exp_func
                new_num_act = num_action + 1
                move = POSSIBLE_MOVE[i]
                action = i
        # Update the number of times we've taken action a' from state s
        # Acquire the alpha first, then update the N
        alpha = decay_constant / (decay_constant + new_num_act - 1)
        set_state_N(curr_state, action, new_num_act)
        # Get the successor state s'
        next_state = curr_state.update_state(move)
        # Perform the TD updates
        max_next_Q = -1
        for i in range(2):
            temp_Q = get_state_Q(next_state, i)
            if temp_Q > max_next_Q:
                max_next_Q = temp_Q
        reward = curr_state.reward
        curr_Q = get_state_Q(curr_state, action)
        new_Q = curr_Q + alpha * (reward + gamma * max_next_Q - curr_Q)
        set_state_Q(curr_state, action, new_Q)
        curr_state = next_state

def agent_move():
    '''
    Agent moves based on training model.
    '''

if __name__ == '__main__':
    learn(0.9, 60.0, 5)