from enum import Enum
import random
import sys

import numpy as np
from scipy.sparse import dia_matrix, block_diag, diags
from scipy.sparse.linalg import eigs
from PIL import Image

# constantes graphiques 
size = width, height = 100, 100
cell_size = 1
gwidth, gheight = int(width / cell_size), int(height / cell_size)

# constantes du modèle
phi_v = 0.04 # densité en vaisseaux sanguins dans la grille
pH_init = 7.4 # pH initial (détermine la répartition d'acide)
kN = 1e-4 # s^(-1) : constante de consommation du glucose par cellules normales (p.7 patel)
kT = 1e-3 # s^(-1) : constante de consommation du glucose par cellules cancéreuses
DG = 9.1e-5 # cm2/s
delta = 2e-3 # 20 µm i.e 2e-3 cm 
q_G = 3.0e-5 # perméabilité de la paroi du vaisseau (cm/s)

# est-ce que certains constantes ne sont pas trops petites/dans la mauvaise unité par rapport à d'autres coefficients ?

pH_D_N, pH_D_T = 6.8, 6.0 # pH seuils qui causent la mort d'une cellule (normale (N), resp. tumeur (T))
pH_Q_N, pH_Q_T = 7.1, 6.4 # pH seuils pour qu'une cellule soit dans un état "quiescent" (calme)

G_D = 2.5 # 2.5 mM, niveau de glucose minimum pour assurer la survie d'une cellule (normale ou cancéreuse confondues)
G_S = 5.0 # mM, niveau par défaut de la répartition de glucose (mM = milimole par litre; autre unité courament utilisée : miligram par décilitre (cf internet pour conversion))

# Section TODO :
# TODO : subdiviser l'ensemble des cellules en 10 groupes aléatoires (qui perdurent pendant toute la durée de la simulation)

# initialisation des tableaux de valeurs
# grid contient l'état de l'automate : type des cellules, active ou non, etc.; élements de type Cell
grid = np.loadtxt("vessels.csv", delimiter=',', dtype=np.int8)
#grid = np.full(shape=(gwidth, gheight), fill_value=3, dtype=np.int8)

# H contient la valeur du pH en tout point
# G contient la concentration en glucose en tout point
# profil de glucose initial : G_S = G_sérum = 5.0 mM partout
H = np.full(shape=(gwidth, gheight), fill_value=pH_init)
G = np.full(shape=(gwidth, gheight), dtype=float, fill_value=G_S)

# répartition aléatoire des vaisseaux dans la grille
def repart_aleat(filename):
    for i in range(width):
        for j in range(height):
            r = random.random()
            if r < phi_v and i > 0 and j > 0 and grid[i-1, j] != 4 and grid[i, j-1] != 4:
                grid[i, j] = 4

    np.savetxt(filename, grid, fmt='%1.0f', delimiter=',')

# méthode de sur-relaxation successive
def k(i, j):
    if grid[i, j] == 2 :
        return kT
    if grid[i, j] == 3:
        return kN
    return 0

# C'est ici que les conditions aux bord de type "périodique" sont mises en place :
# si la case voulue est en dehors de la grille, on recommence au début/à la fin de la ligne/colonne
def voisins(i, j):
    return [grid[i % gwidth, (j+1) % gheight], \
            grid[i % gwidth, (j-1) % gheight], \
            grid[(i+1) % gwidth, j % gheight], \
            grid[(i-1) % gwidth, j % gheight]]

# EMPTY = 0, DEAD = 1,  TUMOR = 2,  NORMAL = 3,  VESSEL = 4
# renvoie la valeur de G à la case i, j affectée du
# bon coefficient selon les cas où on est à côté d'un vaisseau sanguin ou non
def residu(i, j):
    above = (i % gwidth, (j+1) % gheight)
    under = (i % gwidth, (j-1) % gheight)
    right = ((i+1) % gwidth, j % gheight)
    left = ((i-1) % gwidth, j % gheight)
 
    if grid[above] == 4:
        return G[under] + G[left] + G[right] \
                - (3 + ((delta ** 2) * k(i, j) - q_G * delta) / DG) * G[i, j] \
                - ((q_G * delta)/DG) * G_S
    if grid[under] == 4:
        return G[above] + G[left] + G[right] \
                - (3 + ((delta ** 2) * k(i, j) - q_G * delta) / DG) * G[i, j] \
                - ((q_G * delta)/DG) * G_S
    if grid[right] == 4:
        return G[under] + G[left] + G[above] \
                - (3 + ((delta ** 2) * k(i, j) - q_G * delta) / DG) * G[i, j] \
                - ((q_G * delta)/DG) * G_S
    if grid[left] == 4:
        return G[under] + G[above] + G[right] \
                - (3 + ((delta ** 2) * k(i, j) - q_G * delta) / DG) * G[i, j] \
                - ((q_G * delta)/DG) * G_S
    else:
        return G[above] + G[under] + G[left] + G[right] \
                - (4 + (delta ** 2) * k(i, j) / DG) * G[i, j]


# r_jac : rayon spectral de la matrice d'itération utilisée
# dans la méthode de Jacobi lorsque appliquée à ce système
# max_iter : nombre maximal d'itérations avant de s'arrêter
def sor(r_jac, max_iter=100):
    omega = 1.0
    epsilon = 1e-5
    for n in range(1, max_iter):
        norme = 0.0
        max_res_old_fois_omega = 0

        #image = Image.new('RGB', (width, height))

        # la méthode de sur-relaxation précise qu'on modifie omega toutes les demi-étapes;
        # de plus, ça permet d'éviter de mélanger des valeurs nouvelles et des valeurs anciennes
        # dans le calcul des nouvelles valeurs
        # passe une ou passe deux
        for passe in range(1, 3):
            j_debut = passe - 1
            for i in range(0, gwidth):
                for j in range(j_debut, gheight, 2):
                    # on est sur un vaisseau, pas de calculs à faire, la valeur vaut toujours G_S
                    # TODO à reconsidérer ?
                    if grid[i, j] == 4 :
                        #image.putpixel((i, j), (255, 0, 0))
                        pass
                    else:
                        res = residu(i, j)
                        norme += abs(res)
                        diff = omega * res / (4 + ((delta**2)*k(i, j) / DG))
                        # TODO déteminer pourquoi c'est un + et pas un moins (bug, important)
                        # le papier indique un +, le livre Numerical Recipes l'inverse

                        G[i, j] += diff
                        if abs(diff) >= abs(max_res_old_fois_omega):
                            max_res_old_fois_omega = diff


                        #pixelvalue = int((5.5 - G[i, j]) * 255)
                        #image.putpixel((i, j), (pixelvalue, pixelvalue, pixelvalue))

                # j_debut = 1 si 2 avant ou 2 si 1 avant
                j_debut = 1 - j_debut

            if passe == 1 and n == 1:
                omega = 1/(1 - r_jac * r_jac * 0.5)
            else:
                omega = 1/(1 - 0.25 * r_jac * r_jac * omega)

        print(n, " : ", np.amax(G), np.amin(G))
        print(n, " : ", norme)
        print(n, " : ", omega)

        if norme <= epsilon:
            print("norme petite")
            break

        afficher_grille("test-glucose.png")


def afficher_grille(filename):
    image = Image.new('RGB', (width, height))
    for i in range(0, width):
        for j in range(0, height):
            if grid[i, j] == 4:
                image.putpixel((i, j), (255, 255, 255))
            else:
                pixelvalue = int((5.0 - G[i, j]) * 255 * 50)
                image.putpixel((i, j), (0, 0, pixelvalue))

    image.save(filename)

# construction de la matrice éparse du système et détermination du rayon spectral de Jacobi
quatres = np.full(gwidth**2, fill_value=4)
invquatres = np.full(gwidth**2, fill_value=0.25)
uns = np.full(gwidth**2, fill_value=1)

D = dia_matrix((quatres, [0]), shape=(gwidth**2, gwidth**2))
invD = dia_matrix((invquatres, [0]), shape=(gwidth**2, gwidth**2))
L = dia_matrix(([uns, uns], [-1, -gwidth]), shape=(gwidth**2, gwidth**2))
U = dia_matrix(([uns, uns], [1, gwidth]), shape=(gwidth**2, gwidth**2))
A = L + D + U

# Cette matrice est-elle la bonne ?
mat_iter = (-invD) * (L + U)

# calcul du rayon spectral de la matrice (module de la vp la plus grande)
# ce rayon est nécessairement inférieur à 1;
# de plus, plus la grille est grande, plus il se rapproche de 1
rho_jacobi = abs(eigs(mat_iter, k=1, which='LM', return_eigenvectors=False)[0])

print(rho_jacobi)

sor(rho_jacobi, 1000)
