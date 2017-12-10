'''
Main function of 2.1.
'''

import state
import numpy as np
# Possible actions
MOVE_UP = -0.04
MOVE_DOWN = 0.04
GAMMA = 0.9
DECAY_CONSTANT = 60
# State-action array, where value is the Q.
# ball_x, ball_y, vx, vy, paddle_y, gameover, action
STATEACTION = [[0 for i in range(12)], [0 for i in range(12)], [0, 0],
               [0, 0, 0], [0 for i in range(12)], [0, 0], [0, 0]]


def learn(start_state):
    
    pass


if __name__ == '__main__':
    init_state = state.State()