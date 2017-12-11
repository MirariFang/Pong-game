import pygame, sys, pygame.locals
import state
import TrainSmall
import math
pygame.init()
WINDOWS_WIDTH = 400
WINDOWS_HEIGHT = 400
window = pygame.display.set_mode((WINDOWS_WIDTH, WINDOWS_HEIGHT))
pygame.display.set_caption("Pong game")
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PADDLE_YX = WINDOWS_WIDTH - 10


def COORD(_state):
    '''
    Convert a state to ball and paddle coordinates
    '''
    bx = math.floor(_state.ball_x * 400)
    by = math.floor(_state.ball_y * 400)
    cpyr = math.floor(_state.paddle_yr * 400 + 40)
    return bx, by, cpyr


def draw():
    curr_state = state.State()
    ix, iy, cpy = COORD(curr_state)
    line = pygame.Surface((10, 80))
    line.fill(BLACK)
    circle = pygame.Surface((10, 10))
    circle.fill((0, 0, 0))
    pygame.draw.circle(circle, RED, (5, 5), 5, 0)
    circle.set_colorkey((0, 0, 0))
    rects = {'line': line.get_rect(), 'circle': circle.get_rect()}  #24
    rects['line'].centery = cpy
    rects['line'].left = PADDLE_YX
    rects['circle'].centerx = ix
    rects['circle'].centery = iy
    while True:
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()
        curr_state = TrainSmall.get_agent_state(curr_state)
        if curr_state.reward == -1:
            curr_state = state.State()
        c_bx, c_by, c_pyr = COORD(curr_state)
        for rect in rects:
            if rect == 'line':
                rects['line'].centery = c_pyr
                rects['line'].left = PADDLE_YX
            elif rect == 'circle':
                rects['circle'].centerx = c_bx
                rects['circle'].centery = c_by
        window.fill(WHITE)
        window.blit(line, rects['line'])
        window.blit(circle, rects['circle'])
        pygame.time.Clock().tick(FPS)
        pygame.display.update()
