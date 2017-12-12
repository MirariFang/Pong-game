'''
Main function of 2.1. (Small grid)
'''
#TODO: Add statistics of data
import state
import numpy as np
import sys

FLT_MAX = 10000000
# Possible actions
MOVE_UP = -0.04  # index 0
MOVE_DOWN = 0.04  # index 1
NOTHING = 0
POSSIBLE_MOVE = [MOVE_UP, MOVE_DOWN, NOTHING]
#gamma = 0.9
#decay_constant = 60.0
#num_e = 5

# State-action array, where value is the Q.
# Q(s, a) -- ball_x, ball_y, vx, vy, paddle_y, gameover, action
Q_VALUE = np.zeros((24, 24, 4, 5, 24, 2, 3))
# N(s, a)
N_ACTION = np.zeros((24, 24, 4, 5, 24, 2, 3))


def get_state_Q(alter_discrete_state, action):
    '''
    Get the Q-value of the input state.
    '''
    bx = alter_discrete_state[0]
    by = alter_discrete_state[1]
    vx = alter_discrete_state[2]
    vy = alter_discrete_state[3]
    p_yr = alter_discrete_state[4]
    gameover = alter_discrete_state[5]
    return Q_VALUE[bx][by][vx][vy][p_yr][gameover][action]


def set_state_Q(alter_discrete_state, action, value):
    '''
    Set the Q-value of the input state
    '''
    bx = alter_discrete_state[0]
    by = alter_discrete_state[1]
    vx = alter_discrete_state[2]
    vy = alter_discrete_state[3]
    p_yr = alter_discrete_state[4]
    gameover = alter_discrete_state[5]
    Q_VALUE[bx][by][vx][vy][p_yr][gameover][action] = value


def get_state_N(alter_discrete_state, action):
    '''
    Get the N of the input state.
    '''
    bx = alter_discrete_state[0]
    by = alter_discrete_state[1]
    vx = alter_discrete_state[2]
    vy = alter_discrete_state[3]
    p_yr = alter_discrete_state[4]
    gameover = alter_discrete_state[5]
    #print(str(bx) + ' ' + str(by) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(p_yr) + ' ' + str(gameover))
    return N_ACTION[bx][by][vx][vy][p_yr][gameover][action]


def set_state_N(alter_discrete_state, action, value):
    '''
    Set the N of the input state.
    '''
    bx = alter_discrete_state[0]
    by = alter_discrete_state[1]
    vx = alter_discrete_state[2]
    vy = alter_discrete_state[3]
    p_yr = alter_discrete_state[4]
    gameover = alter_discrete_state[5]
    N_ACTION[bx][by][vx][vy][p_yr][gameover][action] = value


def learn(gamma, decay_constant, num_e):
    '''
    A single learning trail.
    '''
    #Q_VALUE[0][0][0][0][0][1][0] = -1
    #Q_VALUE[0][0][0][0][0][1][1] = -1
    curr_state = state.State()
    while 1:
        # From current state s, select an action a.
        curr_state_discrete = curr_state.alter_discrete_state()
        bx = curr_state_discrete[0]
        by = curr_state_discrete[1]
        vx = curr_state_discrete[2]
        vy = curr_state_discrete[3]
        p_yr = curr_state_discrete[4]
        over = curr_state_discrete[5]
        max_exp_func = -FLT_MAX
        action = 0
        new_num_act = 0
        for i in range(3):
            num_action = N_ACTION[bx][by][vx][vy][p_yr][over][i]
            if num_action < num_e:
                exp_func = FLT_MAX
            else:
                exp_func = Q_VALUE[bx][by][vx][vy][p_yr][over][i]
            if exp_func > max_exp_func:
                max_exp_func = exp_func
                new_num_act = num_action + 1
                move = POSSIBLE_MOVE[i]
                action = i
        # Update the number of times we've taken action a' from state s
        # Acquire the alpha first, then update the N
        alpha = decay_constant / (decay_constant + new_num_act - 1)
        N_ACTION[bx][by][vx][vy][p_yr][over][action] = new_num_act
        # Handle the gameover state, just a special case of TD update
        if over == 1:
            over_Q = Q_VALUE[bx][by][vx][vy][p_yr][over][action]
            max_next_Q = -FLT_MAX
            for i in range(3):
                temp_Q = Q_VALUE[bx][by][vx][vy][p_yr][over][i]
                if temp_Q > max_next_Q:
                    max_next_Q = temp_Q
            new_over_Q = over_Q + alpha * (-1 + gamma * max_next_Q - over_Q)
            Q_VALUE[bx][by][vx][vy][p_yr][over][action] = new_over_Q
            break
        # Get the successor state s'
        next_state = curr_state.update_state(move)
        next_state_discrete = next_state.alter_discrete_state()
        nbx = next_state_discrete[0]
        nby = next_state_discrete[1]
        nvx = next_state_discrete[2]
        nvy = next_state_discrete[3]
        np_yr = next_state_discrete[4]
        nover = next_state_discrete[5]
        # Perform the TD updates
        max_next_Q = -FLT_MAX
        for i in range(3):
            temp_Q = Q_VALUE[nbx][nby][nvx][nvy][np_yr][nover][i]
            if temp_Q > max_next_Q:
                max_next_Q = temp_Q
        reward = curr_state.reward
        curr_Q = Q_VALUE[bx][by][vx][vy][p_yr][over][action]
        new_Q = curr_Q + alpha * (reward + gamma * max_next_Q - curr_Q)
        Q_VALUE[bx][by][vx][vy][p_yr][over][action] = new_Q
        curr_state = next_state


def agent_move():
    '''
    Agent moves based on training model.
    '''
    _hit = 0
    curr_state = state.State()
    #count = 0
    while 1:
        #count += 1
        curr_state_discrete = curr_state.alter_discrete_state()
        if curr_state_discrete[5] == 1:
            break
        max_Q = -FLT_MAX
        action = 0
        for i in range(3):
            temp_Q = get_state_Q(curr_state_discrete, i)
            if temp_Q > max_Q:
                max_Q = temp_Q
                action = i
        curr_state = curr_state.update_state(POSSIBLE_MOVE[action])
        if curr_state.reward == 1:
            _hit += 1
    return _hit


def get_agent_state(curr_state):
    '''
    Agent movement function.
    Used for drawing on screen.
    '''
    curr_state_discrete = curr_state.alter_discrete_state()
    if curr_state_discrete[5] == 1:
        return curr_state
    max_Q = -FLT_MAX
    action = 0
    for i in range(3):
        temp_Q = get_state_Q(curr_state_discrete, i)
        if temp_Q > max_Q:
            max_Q = temp_Q
            action = i
    next_state = curr_state.update_state(POSSIBLE_MOVE[action])
    return next_state


def reset_to_0(the_array):
    '''
    Reset the array values to all zeros
    '''
    for i, e in enumerate(the_array):
        if isinstance(e, list):
            reset_to_0(e)
        else:
            the_array[i] = 0


def train(train_num, test_games, gamma, decay_c, num_e):
    '''
    Train a agent with given parameters and training times.
    :return: average number of hits
    '''
    # Initialization, not sure if it's necessary in python
    #Q_VALUE = np.zeros((12, 12, 2, 3, 12, 2, 3))
    #N_ACTION = np.zeros((12, 12, 2, 3, 12, 2, 3))
    reset_to_0(Q_VALUE)
    reset_to_0(N_ACTION)
    print('Start training...', flush=True)
    print(
        'Gamma: %f Decay constant: %d Ne: %d' % (gamma, decay_c, num_e),
        flush=True)
    for i in range(train_num):
        if (i + 1) % 1000 == 0:
            local_total = 0
            for j in range(50):
                local_total += agent_move()
            print(local_total / 50, end=' ', flush=True)
        learn(gamma, decay_c, num_e)
    test_result = test(test_games)
    return test_result


def test(test_games):
    '''
    Use trained model to test.
    '''
    total_hit = 0
    print('Start testing...', flush=True)
    for i in range(test_games):
        #print(i, end=' ', flush=True)
        hit = agent_move()
        total_hit += hit
    return total_hit / test_games


if __name__ == '__main__':
    retval = train(100000, 1000, 0.3, 5, 10)
    print(retval)
