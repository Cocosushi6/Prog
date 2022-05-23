import numpy as np
from PIL import Image
from math import log
from ctes import *

width, height = 100, 100

# diff_cel : est-ce qu'on affiche les différents types de cellules (normal/tumeur/vide)
# en différentes couleurs ou pas
def afficher_grilleH(filename, H, grid, diff_cel=False):
    image = Image.new('RGB', (width, height))
    for i in range(0, width):
        for j in range(0, height):
            pH = - log(H[i, j] * 1e-3, 10)
            pH_S = - log(H_S * 1e-3, 10)

            # correspond à : plus le pH est bas (milieu acide, donc H élevé), plus la couleur est forte
            pixelvalue = abs(int((pH_S - pH) * 125))
            if grid[i, j] == 4:
                image.putpixel((i, j), (255, 255, 255))
            elif grid[i, j] == 3:
                image.putpixel((i, j), (0, pixelvalue, 0))
            elif grid[i, j] == 2:
                if diff_cel:
                    image.putpixel((i, j), (pixelvalue, pixelvalue, 0))
                else:
                    image.putpixel((i, j), (0, pixelvalue, 0))
            elif grid[i, j] == 0:
                if diff_cel:
                    image.putpixel((i, j), (pixelvalue, pixelvalue, pixelvalue))
                else:
                    image.putpixel((i, j), (0, pixelvalue, 0))

    image.save(filename)

# diff_cel : est-ce qu'on affiche les différents types de cellules (normal/tumeur/vide)
# en différentes couleurs ou pas
def afficher_grilleG(filename, G, grid, diff_cel=False):
    image = Image.new('RGB', (width, height))
    for i in range(0, width):
        for j in range(0, height):
            pixelvalue = abs(255 - int((G_S - G[i, j]) * 255 * 40)) # valeurs magiques
            if grid[i, j] == 4:
                image.putpixel((i, j), (255, 255, 255))
            elif grid[i, j] == 3:
                image.putpixel((i, j), (0, 0, pixelvalue))
            elif grid[i, j] == 2:
                if diff_cel:
                    image.putpixel((i, j), (pixelvalue, 0, 0))
                else:
                    image.putpixel((i, j), (0, 0, pixelvalue))
            elif grid[i, j] == 0:
                if diff_cel:
                    image.putpixel((i, j), (pixelvalue, pixelvalue, pixelvalue))
                else:
                    image.putpixel((i, j), (0, 0, pixelvalue))


    image.save(filename)

def afficher_etat(filename, E, grid):
    image = Image.new('RGB', (width, height))
    for i in range(0, width):
        for j in range(0, height):
            if grid[i, j] == 4:
                image.putpixel((i, j), (255, 255, 255))
            elif grid[i, j] == 3:
                if E[i, j] == 1:
                    image.putpixel((i, j), (0, 0, 255))
                else:
                    image.putpixel((i, j), (0, 0, 175))
            elif grid[i, j] == 2:
                if E[i, j] == 1:
                    image.putpixel((i, j), (255, 0, 0))
                else:
                    image.putpixel((i, j), (175, 0, 0))

    image.save(filename)

