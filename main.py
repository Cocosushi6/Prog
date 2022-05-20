from enum import Enum
import random
import sys

import numpy as np
from scipy.sparse import dia_matrix, block_diag, diags
from scipy.sparse.linalg import eigs
from PIL import Image

# Section TODO :
# TODO : subdiviser l'ensemble des cellules en 10 groupes aléatoires (qui perdurent pendant toute la durée de la simulation)

# constantes graphiques 
size = width, height = 100, 100

# constantes du modèle
phi_v = 0.04 # densité en vaisseaux sanguins dans la grille
kN = 1e-4 # s^(-1) : constante de consommation du glucose par cellules normales (p.7 patel)
kT = 1e-3 # s^(-1) : constante de consommation du glucose par cellules cancéreuses
DG = 9.1e-5 # cm2/s
DH = 1.08e-5 #
delta = 2e-3 # 20 µm i.e 2e-3 cm 
q_G = 3.0e-5 # perméabilité de la paroi du vaisseau (cm/s) au glucose
q_H = 1.19e-4 # pareil, mais aux ions H+ (cm/s)

# est-ce que certains constantes ne sont pas trops petites/dans la mauvaise unité par rapport à d'autres coefficients ?

H_D_N, pH_D_T = 6.8, 6.0 # pH seuils qui causent la mort d'une cellule (normale (N), resp. tumeur (T))
pH_Q_N, pH_Q_T = 7.1, 6.4 # pH seuils pour qu'une cellule soit dans un état "quiescent" (calme)

H_A, H_Q = 5e-5, 5e-7 # taux de production d'acide des cellules cancéreuses, en mM/s (milimoles par litre par secondes)

G_D = 2.5 # 2.5 mM, niveau de glucose minimum pour assurer la survie d'une cellule (normale ou cancéreuse confondues)
G_S = 5.0 # mM, niveau par défaut de la répartition de glucose (mM = milimole par litre; autre unité courament utilisée : miligram par décilitre (cf internet pour conversion))
H_S = 3.98e-5 # concentration en ions H+ du sérum/sang (i.e pH = 7.4)


# initialisation des tableaux de valeurs
# grid contient l'état de l'automate : type des cellules, active ou non, etc.; élements de type Cell
#grid = np.full(shape=(width, height), fill_value=3, dtype=np.int8)
grid = np.loadtxt("vessels.csv", delimiter=',', dtype=np.int8)

# H contient la valeur du pH en tout point
# G contient la concentration en glucose en tout point
# profil de glucose initial : G_S = G_sérum = 5.0 mM partout
# E contient l'Etat de chaque cellule : 0 pour endormie/quiet, 1 pour active
H = np.full(shape=(width, height), fill_value=H_S)
G = np.full(shape=(width, height), dtype=float, fill_value=G_S)
E = np.zeros(shape=(width, height), dtype=np.int8)

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
    epsilon = 1e-5
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
                    # la discussion dans Patel et al. est juste pour établir la bonne expression, car on ne
                    # peut pas juste balancer G_ij = G_S comme ça.
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

        afficher_grilleG("Images/test" + str(n) + ".png")
    np.savetxt("donnes.csv", tab, fmt='%1.6f', delimiter=',')
   
def afficher_grilleH(filename):
    image = Image.new('RGB', (width, height))
    for i in range(0, width):
        for j in range(0, height):
            if grid[i, j] == 4:
                image.putpixel((i, j), (255, 255, 255))
            elif grid[i, j] == 3:
                pixelvalue = abs(255 - int((H_S - H[i, j]) * 255 * 50 * (255/150)))
                image.putpixel((i, j), (0, 0, pixelvalue))
            elif grid[i, j] == 2:
                pixelvalue = abs(255 - int((H_S - H[i, j]) * 255 * 50 * (255/150)))
                image.putpixel((i, j), (pixelvalue, 0, 0))

    image.save(filename)

def afficher_grilleG(filename):
    image = Image.new('RGB', (width, height))
    for i in range(0, width):
        for j in range(0, height):
            if grid[i, j] == 4:
                image.putpixel((i, j), (255, 255, 255))
            elif grid[i, j] == 3:
                pixelvalue = abs(255 - int((5.0 - G[i, j]) * 255 * 50 * (255/150)))
                image.putpixel((i, j), (0, 0, pixelvalue))
            elif grid[i, j] == 2:
                pixelvalue = abs(255 - int((5.0 - G[i, j]) * 255 * 50 * (255/150)))
                image.putpixel((i, j), (pixelvalue, 0, 0))

    image.save(filename)

# N : nombre d'itérations de l'automate
def automate(max_iter):
    # préparation de l'automate
    # construction de la matrice éparse du système et détermination du rayon spectral de Jacobi
    quarts = np.full(width**2, fill_value=0.25)
    uns = np.full(width**2, fill_value=1)

    invD = dia_matrix((quarts, [0]), shape=(width**2, width**2))
    L = dia_matrix(([uns, uns], [-1, -width]), shape=(width**2, width**2))
    U = dia_matrix(([uns, uns], [1, width]), shape=(width**2, width**2))

    # Cette matrice est-elle la bonne ?
    mat_iter = (-invD) * (L + U)

    # calcul du rayon spectral de la matrice (module de la vp la plus grande)
    # ce rayon est nécessairement inférieur à 1;
    # de plus, plus la grille est grande, plus il se rapproche de 1
    rho_jacobi = abs(eigs(mat_iter, k=1, which='LM', return_eigenvectors=False)[0])
    print(rho_jacobi)

    sor(rho_jacobi, G, residuG, 1000)
    sor(rho_jacobi, H, residuH, 1000)

    for N in range(0, max_iter):
        for i in range(0, width):
            for j in range(0, height):
                # EMPTY = 0, DEAD = 1, TUMOR = 2, NORMAL = 3, VESSEL = 4
                pass
        pass

automate(20)
