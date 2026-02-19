import pygame
import math
import random

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Thermonuclear Missile Simulation")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FIRE = (255, 100, 0)
SHOCKWAVE = (255, 200, 200)
RADIATION = (150, 255, 150)

# Missile setup
missile_pos = [100, HEIGHT - 50]
target = [700, HEIGHT - 50]
missile_radius = 5
exploded = False
explosion_radius = 0
gravity = 0.2

# Calculate launch velocity
angle = math.radians(45)
speed = 10
velocity = [math.cos(angle) * speed, -math.sin(angle) * speed]

# Main loop
running = True
while running:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not exploded:
        # Draw missile
        pygame.draw.circle(screen, WHITE, (int(missile_pos[0]), int(missile_pos[1])), missile_radius)

        # Update position
        missile_pos[0] += velocity[0]
        missile_pos[1] += velocity[1]
        velocity[1] += gravity

        # Detect impact
        if missile_pos[1] >= HEIGHT - 50:
            exploded = True
            explosion_center = missile_pos[:]
    else:
        # Explosion visuals
        if explosion_radius < 200:
            explosion_radius += 3
            pygame.draw.circle(screen, FIRE, (int(explosion_center[0]), int(explosion_center[1])), explosion_radius)
            pygame.draw.circle(screen, SHOCKWAVE, (int(explosion_center[0]), int(explosion_center[1])), explosion_radius//2, 3)
            pygame.draw.circle(screen, RADIATION, (int(explosion_center[0]), int(explosion_center[1])), explosion_radius//3, 1)
        else:
            font = pygame.font.SysFont(None, 40)
            text = font.render("Target Obliterated", True, WHITE)
            screen.blit(text, (WIDTH//2 - 100, HEIGHT//2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

