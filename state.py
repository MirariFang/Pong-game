'''
Frame of the pong game.
'''
import random
import math
PADDLE_HEIGHT = 0.2
REWARD = 0


class State:
    'Class that contains all attributes of the world state.'

    def __init__(self,
                 ball_x=0.5,
                 ball_y=0.5,
                 velocity_x=0.03,
                 velocity_y=0.01,
                 paddle_yr=0.5 - PADDLE_HEIGHT / 2,
                 reward=0):
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.paddle_yr = paddle_yr
        self.reward = reward

    def cont_state(self):
        '''
        Return the current state (continuous).
        '''
        return [
            self.ball_x, self.ball_y, self.velocity_x, self.velocity_y,
            self.paddle_yr
        ]

    def discrete_state(self):
        '''
        Return the current state (discrete).
        This is used for learning.
        '''
        curr_ball_x = math.floor(self.ball_x * 12 - 1)
        curr_ball_y = math.floor(self.ball_y * 12 - 1)
        curr_vx = 1 if self.velocity_x > 0 else -1
        if self.velocity_y >= 0.015:
            curr_vy = 1
        elif self.velocity_y <= -0.015:
            curr_vy = -1
        else:
            curr_vy = 0
        if self.paddle_yr == 1 - PADDLE_HEIGHT:
            discrete_paddle_yr = 11
        else:
            discrete_paddle_yr = math.floor(12 * self.paddle_yr /
                                            (1 - PADDLE_HEIGHT))
        if self.ball_x > 1 and self.reward == -1:
            gameover = 1
        else:
            gameover = 0
        return [
            curr_ball_x, curr_ball_y,
            int((curr_vx + 1) / 2),
            int((curr_vy + 1) / 2), discrete_paddle_yr, gameover
        ]

    def alter_discrete_state(self):
        '''
        A discrete state with a different "resolution".
        '''
        curr_ball_x = math.floor(self.ball_x * 24 - 1)
        curr_ball_y = math.floor(self.ball_y * 24 - 1)
        if self.velocity_x > 0:
            if self.velocity_x > 0.4:
                curr_vx = 3
            else:
                curr_vx = 2
        else:
            if self.velocity_x < 0.4:
                curr_vx = 0
            else:
                curr_vx = 1
        if self.velocity_y >= 0.015:
            if self.velocity_y > 0.1:
                curr_vy = 4
            else:
                curr_vy = 3
        elif self.velocity_y <= -0.015:
            if self.velocity_y < -0.1:
                curr_vy = 0
            else:
                curr_vy = 1
        else:
            curr_vy = 2
        if self.paddle_yr == 1 - PADDLE_HEIGHT:
            discrete_paddle_yr = 23
        else:
            discrete_paddle_yr = math.floor(24 * self.paddle_yr /
                                            (1 - PADDLE_HEIGHT))
        if self.ball_x > 1 and self.reward == -1:
            gameover = 1
        else:
            gameover = 0
        return [
            curr_ball_x, curr_ball_y, curr_vx, curr_vy, discrete_paddle_yr,
            gameover
        ]

    def reset(self):
        '''
        Reset the world state.
        '''
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.velocity_x = 0.03
        self.velocity_y = 0.01
        self.paddle_yr = 0.5 - PADDLE_HEIGHT / 2
        self.reward = 0

    def update_state(self, action):
        '''
        Continuously Update the state.
        :param action: the action of the paddle, can be 0.04 (goes down) or -0.04 (goes up).
        :return: return the next state
        '''
        self.reward = 0
        next_state = State()
        next_state.velocity_x = self.velocity_x
        next_state.velocity_y = self.velocity_y
        next_state.paddle_yr = self.paddle_yr + action
        if next_state.paddle_yr < 0:
            next_state.paddle_yr = 0
        elif next_state.paddle_yr > 1 - PADDLE_HEIGHT:
            next_state.paddle_yr = PADDLE_HEIGHT
        # Update ball position
        next_state.ball_x = self.ball_x + self.velocity_x
        next_state.ball_y = self.ball_y + self.velocity_y
        # Check for bouncing
        if next_state.ball_y < 0:
            next_state.ball_y = -self.ball_y
            next_state.velocity_y = -self.velocity_y
        if next_state.ball_y > 1:
            next_state.ball_y = 2 - self.ball_y
            next_state.velocity_y = -self.velocity_y
        if next_state.ball_x < 0:
            next_state.ball_x = -self.ball_x
            next_state.velocity_x = -self.velocity_x
        # Check for paddle bouncing
        if next_state.ball_x > 1:
            if next_state.ball_y >= next_state.paddle_yr and next_state.ball_y <= (
                    next_state.paddle_yr + PADDLE_HEIGHT):
                next_state.reward = 1
                u = random.uniform(-0.015, 0.015)
                v = random.uniform(-0.03, 0.03)
                new_vx = -abs(self.velocity_x + u)
                next_state.ball_x = 2 - next_state.ball_x
                if abs(new_vx) > 0.03:
                    next_state.velocity_x = new_vx
                else:
                    next_state.velocity_x = -0.03
                next_state.velocity_y = self.velocity_y + v
            else:
                next_state.reward = -1  # Out of bound
                # self.reset()
        # Restrict maximum v_x and v_y
        # Hardcode them to 0.3 or -0.3
        if abs(next_state.velocity_x) > 1:
            if next_state.velocity_x > 0:
                next_state.velocity_x = 0.3
            else:
                next_state.velocity_x = -0.3
            #next_state.velocity_x = old_vx + random.uniform(-0.015, 0.015)
        if abs(next_state.velocity_y) > 1:
            if next_state.velocity_y > 0:
                next_state.velocity_y = 0.3
            else:
                next_state.velocity_y = -0.3
            #next_state.velocity_y = old_vy + random.uniform(-0.03, 0.03)
        return next_state

    def print_state(self):
        '''
        Print current state. For debugging purpose.
        '''
        print(self.ball_x, end=' ')
        print(self.ball_y, end=' ')
        print(self.velocity_x, end=' ')
        print(self.velocity_y, end=' ')
        print(self.paddle_yr, end=' ')
        print(self.reward)


class TwoPaddleState(State):
    'Class that contains the world state of 2 paddles game'

    def __init__(self, paddle_yl=0.5 - PADDLE_HEIGHT / 2):
        State.__init__(self)
        self.paddle_yl = paddle_yl

    #TODO: Your codes go here!!!