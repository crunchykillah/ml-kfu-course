import pygame
import numpy as np

# DBSCAN algorithm
def dbscan(D, eps, MinPts):
    labels = [0]*len(D)
    C = 0

    for P in range(0, len(D)):
        if not (labels[P] == 0):
           continue

        NeighborPts = regionQuery(D, P, eps)

        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else:
           C += 1
           growCluster(D, labels, P, NeighborPts, C, eps, MinPts)

    return labels

def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C

    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]

        if labels[Pn] == -1:
           labels[Pn] = C

        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = regionQuery(D, Pn, eps)

            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts

        i += 1

def regionQuery(D, P, eps):
    neighbors = []

    for Pn in range(0, len(D)):
        if np.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)

    return neighbors

# Pygame visualization
pygame.init()
screen = pygame.display.set_mode((800, 600))

points = []
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            points.append(pygame.mouse.get_pos())

    screen.fill((0, 0, 0))
    for point in points:
        pygame.draw.circle(screen, (255, 255, 255), point, 5)
    pygame.display.flip()

pygame.quit()

# Run DBSCAN
labels = dbscan(np.array(points), eps=100, MinPts=5)

# Color the points based on the labels
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
pygame.init()
screen = pygame.display.set_mode((800, 600))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))
    for i, point in enumerate(points):
        pygame.draw.circle(screen, colors[labels[i]], point, 5)
    pygame.display.flip()

pygame.quit()
