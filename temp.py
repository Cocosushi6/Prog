import numpy as np
from main import *

# pour l'instant, version non indépendante : 
# les cellules de la sous-génération suivante vont dépendre 
# des résultats de la sous-génération précédente : 
# - grid va être modifié avec les nouvelles cellules de la génération 
#   précédente
# - E va être modifié avec les mises en sommeil/activité de 
#   la génération précédente
# - sor va être appelé avec la grille modifié; en retour, G et H 
#   seront changé à chaque sous-génération en fonction des précédentes
partition = split(grid)
for l in range(len(partition)):
    newgrid = grid.copy()
    x, y = np.nonzero(partition[l])    

    print("SUBSTEP : ", str(l))
    print("Taille partition : ", len(x))

    for count in range(len(x)):
        i, j = x[count], y[count]

        # EMPTY = 0, DEAD = 1, TUMOR = 2, NORMAL = 3, VESSEL = 4
        if grid[i, j] == 2 or grid[i, j] == 3:
            pH = -log(H[i, j] * 1e-3, 10) # H est en milimoles/L donc on reconverti en mol/L pour avoir le pH

            # si le pH ou la concentration en glucose trop basses : mort de la cellule
            if (pH <= (pH_D_N if grid[i, j] == 3 else pH_D_T) or G[i, j] <= 2.5) :
                newgrid[i, j] = 1
                E[i, j] = 0
                print("mort cellule : ", (i, j))

            # sinon, si entre les deux, état de "sommeil"
            # SOMMEIL = 0, ACTIF = 1
            elif pH <= (pH_Q_N if grid[i, j] == 3 else pH_Q_T):
                E[i, j] = 0
                print("mise en sommeil : ", (i, j))

            # enfin, si plus haut : division et reproduction de la cellule
            else:
                # le pH est + haut que les seuils, on réactive la cellule
                E[i, j] = 1
                # on crée une nouvelle cellule si possible
                if alent_vacants(i, j):
                    # TODO remplacer ça par un truc aléatoire si rien ne fonctionne
                    x1, y1 = max_glucose(i, j)
                    newgrid[x1 % width, y1 % height] = grid[i, j]
                    # on regarde si elle est immédiatement en sommeil ou non
                    # TODO ici pH devrait pas être évalué en i, j mais à l'endroit de la nouvelle cellule
                    # TODO E devrait être séparé en E et new_E comme pour grid, pour éviter les fausses corrélations
                    if pH <= (pH_Q_N if newgrid[x1 % width, y1 % height] == 3 else pH_Q_T):
                        E[x1 % width, y1 % height] = 0
                    else:
                        E[x1 % width, y1 % height] = 1

    grid = newgrid.copy()

    sor(rho_jacobi, grid, G, residuG, x, y, 1000)
    print("SOR G fini")

    sor(rho_jacobi, grid, H, residuH, x, y, 1000)
    print("SOR H fini")
 
    #afficher_grilleG("Images/testG" + str(N) + "-" + str(l) + ".png", G, grid)
    #np.savetxt("Data/donnesG" + str(N) + "-" + str(l) + ".csv", G, fmt='%1.5f', delimiter=',')
    #afficher_grilleH("Images/testH" + str(N) + "-" + str(l) + ".png", H, grid) 
    #np.savetxt("Data/donnesH" + str(N) + "-" + str(l) + ".csv", H, fmt='%1.9f', delimiter=',')
    #afficher_etat("Images/etat" + str(N) + "-" + str(l) + ".png", E, grid)
    #np.savetxt("Data/etat" + str(N) + "-" + str(l) + ".csv", E, fmt='%1.0f', delimiter=',')
    #np.savetxt("Data/type" + str(N) + "-" + str(l) + ".csv", grid, fmt='%1.0f', delimiter=',')
