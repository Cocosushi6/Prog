from enum import Enum
from math import log
import random
import sys

import numpy as np
from scipy.sparse import dia_matrix, block_diag, diags
from scipy.sparse.linalg import eigs
from PIL import Image

from ctes import *
from affichage import *

# Section TODO :
# TODO : subdiviser l'ensemble des cellules en 10 groupes aléatoires (qui perdurent pendant toute la durée de la simulation)

# initialisation des tableaux de valeurs
# grid contient l'état de l'automate : type des cellules, active ou non, etc.; élements de type Cell
#grid = np.full(shape=(width, height), fill_value=3, dtype=np.int8)
grid = np.loadtxt("vessels-noT.csv", delimiter=',', dtype=np.int8)
#repart_aleat("vessels.csv")

# H contient la valeur du pH en tout point
# G contient la concentration en glucose en tout point
# profil de glucose initial : G_S = G_sérum = 5.0 mM partout
# E contient l'Etat de chaque cellule : 0 pour endormie/quiet, 1 pour active
H = np.full(shape=(width, height), fill_value=H_S)
G = np.full(shape=(width, height), dtype=float, fill_value=G_S)
E = np.ones(shape=(width, height), dtype=np.int8)

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

def h(i, j):
    if grid[i, j] == 2:
        return H_A if E[i, j] == 1 else H_Q
    return 0

def Gl(i, j):
    return G[i % width, j % height]

def Grid(i, j):
    return grid[i % width, j % height]

def residuH(i, j):
    above = (i % width, (j+1) % height)
    under = (i % width, (j-1) % height)
    right = ((i+1) % width, j % height)
    left = ((i-1) % width, j % height)

    if grid[above] == 4:
        return (H[under] + H[left] + H[right] \
                - (3 + q_H * delta / DH) * H[i, j] \
                + ((q_H * delta)/DH) * H_S \
                + (delta ** 2) / DH * h(i, j), (3 + q_H * delta / DH))
    if grid[under] == 4:
        return (H[above] + H[left] + H[right] \
                - (3 + (q_H * delta) / DH) * H[i, j] \
                + ((q_H * delta)/DH) * H_S \
                + (delta ** 2) / DH * h(i, j), (3 + q_H * delta / DH))
    if grid[right] == 4:
        return (H[under] + H[left] + H[above] \
                - (3 + (q_H * delta) / DH) * H[i, j] \
                + ((q_H * delta)/DH) * H_S \
                + (delta ** 2) / DH * h(i, j), (3 + q_H * delta / DH))
    if grid[left] == 4:
        return (H[under] + H[above] + H[right] \
                - (3 + (q_H * delta) / DH) * H[i, j] \
                + ((q_H * delta)/DH) * H_S \
                + (delta ** 2) / DH * h(i, j), (3 + q_H * delta / DH))
    else:
        return (H[above] + H[under] + H[left] + H[right] \
                - 4 * H[i, j] \
                + (delta ** 2 / DH) * h(i, j), 4)



# EMPTY = 0, DEAD = 1, TUMOR = 2, NORMAL = 3, VESSEL = 4
# renvoie la valeur de G à la case i, j affectée du
# bon coefficient selon les cas où on est à côté d'un vaisseau sanguin ou non
def residuG(i, j):
    above = (i % width, (j+1) % height)
    under = (i % width, (j-1) % height)
    right = ((i+1) % width, j % height)
    left = ((i-1) % width, j % height)

    if grid[above] == 4:
        return (G[under] + G[left] + G[right] \
                - (3 + ((delta ** 2) * k(i, j) + q_G * delta) / DG) * G[i, j] \
                + ((q_G * delta)/DG) * G_S, (3 + ((delta ** 2) * k(i, j) + q_G * delta) / DG))
    if grid[under] == 4:
        return (G[above] + G[left] + G[right] \
                - (3 + ((delta ** 2) * k(i, j) + q_G * delta) / DG) * G[i, j] \
                + ((q_G * delta)/DG) * G_S, (3 + ((delta ** 2) * k(i, j) + q_G * delta) / DG))
    if grid[right] == 4:
        return (G[under] + G[left] + G[above] \
                - (3 + ((delta ** 2) * k(i, j) + q_G * delta) / DG) * G[i, j] \
                + ((q_G * delta)/DG) * G_S, (3 + ((delta ** 2) * k(i, j) + q_G * delta) / DG))
    if grid[left] == 4:
        return (G[under] + G[above] + G[right] \
                - (3 + ((delta ** 2) * k(i, j) + q_G * delta) / DG) * G[i, j] \
                + ((q_G * delta)/DG) * G_S, (3 + ((delta ** 2) * k(i, j) + q_G * delta) / DG))
    else:
        return (G[above] + G[under] + G[left] + G[right] \
                - (4 + (delta ** 2) * k(i, j) / DG) * G[i, j], (4 + ((delta**2)*k(i, j) / DG)))


# r_jac : rayon spectral de la matrice d'itération utilisée
# dans la méthode de Jacobi lorsque appliquée à ce système
# max_iter : nombre maximal d'itérations avant de s'arrêter
def sor(r_jac, tab, f_residu, max_iter=100):
    omega = 1.0
    epsilon = 1e-4
    for n in range(1, max_iter):
        norme = 0.0
        max_res_old_fois_omega = 0

        # la méthode de sur-relaxation précise qu'on modifie omega toutes les demi-étapes;
        # de plus, ça permet d'éviter de mélanger des valeurs nouvelles et des valeurs anciennes
        # dans le calcul des nouvelles valeurs
        # passe une ou passe deux
        for passe in range(1, 3):
            j_debut = passe - 1
            for i in range(0, width):
                for j in range(j_debut, height, 2):
                    # on est sur un vaisseau, pas de calculs à faire, la valeur vaut toujours G_S
                    # TODO à reconsidérer ?
                    # normalement c'est bon; dans un vaisseau, la concentration est toujours G_S;
                    # la discussion dans Patel et al. est juste pour établir la bonne expression,
                    # car on ne peut pas juste balancer G_ij = G_S comme ça.
                    if grid[i, j] == 4 :
                        pass
                    else:
                        res, coef = f_residu(i, j)
                        norme += abs(res)
                        diff = omega * res / coef

                        # TODO déteminer pourquoi c'est un + et pas un moins (bug, important)
                        # le papier indique un +, le livre Numerical Recipes l'inverse

                        tab[i, j] += diff
                        if abs(diff) >= abs(max_res_old_fois_omega):
                            max_res_old_fois_omega = diff

                # j_debut = 1 si 2 avant ou 2 si 1 avant
                j_debut = 1 - j_debut

            if passe == 1 and n == 1:
                omega = 1/(1 - r_jac * r_jac * 0.5)
            else:
                omega = 1/(1 - 0.25 * r_jac * r_jac * omega)

        print(n, " : ", np.amax(tab), np.amin(tab))
        print(n, " : ", norme)
        print(n, " : ", omega)

        if norme <= epsilon:
            print("norme petite")
            break

#        afficher_grilleH("Images/test" + str(n) + ".png")
#    np.savetxt("donnes.csv", tab, fmt='%1.6f', delimiter=',')

def alent_vacants(i, j):
    return (Grid(i+1, j) == 0 or Grid(i-1, j) == 0 or Grid(i, j+1) == 0 or Grid(i, j-1) == 0)

def max_glucose(i, j):
    d = {(i+1, j): Gl(i+1, j), \
         (i-1, j): Gl(i-1, j), \
         (i, j+1): Gl(i, j+1), \
         (i, j-1): Gl(i, j-1)}
    return max(d, key=d.get) # cf internet

# N : nombre d'itérations de l'automate
def automate(rho_jacobi, max_iter):
    sor(rho_jacobi, G, residuG, 1000)
    afficher_grilleG("Images/testG.png", G, grid)
    np.savetxt("Data/donnesG.csv", G, fmt='%1.5f', delimiter=',')

    sor(rho_jacobi, H, residuH, 1000)
    afficher_grilleH("Images/testH.png", H, grid)
    np.savetxt("Data/donnesH.csv", H, fmt='%1.9f', delimiter=',')

    afficher_etat("Images/etat.png", E, grid)
    np.savetxt("Data/etat.csv", E, fmt='%1.0f', delimiter=',')
    np.savetxt("Data/type.csv", grid, fmt='%1.0f', delimiter=',')

    for N in range(0, max_iter):
        for i in range(0, width):
            for j in range(0, height):
                # EMPTY = 0, DEAD = 1, TUMOR = 2, NORMAL = 3, VESSEL = 4
                if grid[i, j] == 2 or grid[i, j] == 3:
                    pH = -log(H[i, j] * 1e-3, 10) # H est en milimoles/L donc on reconverti en mol/L pour avoir le pH

                    # si le pH ou la concentration en glucose trop basses : mort de la cellule
                    if pH <= (pH_D_N if grid[i, j] == 3 else pH_D_T) \
                     or G[i, j] <= 2.5 :
                        grid[i, j] = 0
                        print("mort cellule : ", (i, j))

                    # sinon, si entre les deux, état de "sommeil"
                    elif pH <= (pH_Q_N if grid[i, j] == 3 else pH_Q_T):
                        E[i, j] = 0
                        print("mise en sommeil : ", (i, j))

                    # enfin, si plus haut : division et reproduction de la cellule
                    else:
                        if alent_vacants(i, j):
                            x, y = max_glucose(i, j)
                            grid[x % width, y % height] = grid[i, j]

        print("étape : ", str(N))

        sor(rho_jacobi, G, residuG, 1000)
        afficher_grilleG("Images/testG" + str(N) + ".png", G, grid) 
        np.savetxt("Data/donnesG" + str(N) + ".csv", G, fmt='%1.5f', delimiter=',')
        print("SOR G fini")

        sor(rho_jacobi, H, residuH, 1000)
        afficher_grilleH("Images/testH" + str(N) + ".png", H, grid) 
        np.savetxt("Data/donnesH" + str(N) + ".csv", H, fmt='%1.9f', delimiter=',')
        print("SOR H fini")
        
        afficher_etat("Images/etat" + str(N) + ".png", E, grid)
        np.savetxt("Data/etat" + str(N) + ".csv", E, fmt='%1.0f', delimiter=',')

        np.savetxt("Data/type" + str(N) + ".csv", grid, fmt='%1.0f', delimiter=',')

# préparation de l'automate
# construction de la matrice éparse du système et détermination du rayon spectral de Jacobi
quarts = np.full(width*height, fill_value=0.25)
uns = np.full(width*height, fill_value=1)

invD = dia_matrix((quarts, [0]), shape=(width**2, height**2))
L = dia_matrix(([uns, uns], [-1, -width]), shape=(width**2, height**2))
U = dia_matrix(([uns, uns], [1, width]), shape=(width**2, height**2))

# Cette matrice est-elle la bonne ?
mat_iter = (-invD) * (L + U)

# calcul du rayon spectral de la matrice (module de la vp la plus grande)
# ce rayon est nécessairement inférieur à 1;
# de plus, plus la grille est grande, plus il se rapproche de 1
rho_jacobi = abs(eigs(mat_iter, k=1, which='LM', return_eigenvectors=False)[0])
print(rho_jacobi)

#sor(rho_jacobi, H, residuH, 1000)

automate(rho_jacobi, 20)
