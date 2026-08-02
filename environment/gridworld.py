import pygame
import sys

# -----------------------
# Configuration
# -----------------------

GRID_SIZE = 5
CELL_SIZE = 100

WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (40,120,255)
GREEN = (0,200,0)

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GridWorld")

clock = pygame.time.Clock()

agent = [0,0]
goal = [4,4]

# -----------------------
# Main Loop
# -----------------------

while True:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        agent[0] = max(0, agent[0]-1)

    if keys[pygame.K_DOWN]:
        agent[0] = min(GRID_SIZE-1, agent[0]+1)

    if keys[pygame.K_LEFT]:
        agent[1] = max(0, agent[1]-1)

    if keys[pygame.K_RIGHT]:
        agent[1] = min(GRID_SIZE-1, agent[1]+1)

    screen.fill(WHITE)

    # draw grid

    for i in range(GRID_SIZE):

        pygame.draw.line(screen,BLACK,(0,i*CELL_SIZE),(WIDTH,i*CELL_SIZE))
        pygame.draw.line(screen,BLACK,(i*CELL_SIZE,0),(i*CELL_SIZE,HEIGHT))

    # goal

    pygame.draw.rect(
        screen,
        GREEN,
        (
            goal[1]*CELL_SIZE,
            goal[0]*CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
    )

    # agent

    pygame.draw.circle(
        screen,
        BLUE,
        (
            agent[1]*CELL_SIZE + CELL_SIZE//2,
            agent[0]*CELL_SIZE + CELL_SIZE//2
        ),
        25
    )

    pygame.display.flip()
    clock.tick(10)