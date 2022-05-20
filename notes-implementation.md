Notes sur l'implémentation de la méthode SOR (successive overrelaxation)
- dans notre problème, les conditions aux bords sont périodiques (normal, on travaille sur une petite portion de tissu humain) : ainsi pour tout i, j, u(i, 0) = u(i, J), u(0, j) = u(I, j)
	- i.e quand on calcule u(i, 0) par exemple, l'une des quatres valeurs nécessaires est hors de portée : u(i, -1) hors de la grille, donc on prend u(i, J) (wrap)

- le paramètre de relaxation optimal est w = 2/(1 + sqrt(1 - rho_jacobi ^ 2)), et rho_jacobi est le rayon spectral de la matrice d'itération de Jacobi : 
	- comment déterminer ce rayon spectral dans notre cas : 
		- les solutions analytiques données dans la littérature ne sont valables que pour des conditions aux bord dites "de Dirichlet", où u = 0; ce n'est pas notre cas : 
			- on peut toujours essayer de l'adapter, car l'équation de base correspond plutôt bien à celle pris en exemple dans Numerical Recipes, même si l'ajout de vaisseaux sanguins   fausse l'équivalence  
			- en plus, les *conditions aux bords* ne sont pas les mêmes (Dirichlet vs périodique)
		- sinon, on peut utiliser (mais c'est beaucoup plus couteux) la définition du rayon spectral : c'est le maximum des valeurs propres de cette matrice; on peut donc essayer de déterminer une à une les valeurs propres, voire déterminer seulement les plus hautes ? 
		- cf module python scipy, qui peut permettre de déterminer les valeurs propres d'une matrice, même si il semblerait qu'il y ait quelques problèmes de précision
			- https://stackoverflow.com/questions/30966881/python-scipy-sparse-linalg-eigs-giving-different-results-for-consecutive-calls
			- https://stackoverflow.com/questions/19992255/finding-spectral-radius-of-the-jacobi-iteration-matrix (utilise matlab)
		- 
	
	- stockage de la matrice de multiplication (A) : il faut utiliser scipy.sparse pour éviter de stocker trop de zéros (la matrice est /éparse/, dispersée, peu dense)
		- utiliser la classe dia_matrix avec le constructeur dia_matrix((data, offset), shape) : voir exemple là : https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.sparse.dia_matrix.html#scipy.sparse.dia_matrix 
		- permet de créér une matrice éparse sans trop de données (offset sera un petit tableau dans notre cas)
		- 

	- valeur initiale du champ de Glucose G : le prendre à 0, et regarder ce que ça donne
	- conditions aux bords des vaisseaux sanguins : le papier précise que cela implique seulement que certains coefficients dans notre matrice vont changerprobablement 
	- "source" de l'équation (le rho dans Numerical Recipes) : ici G (le champ de concentration en glucose) est sa propre source d'après l'équation qu'ils utilisent 

	- commencer par implémenter la méthode de sur-relaxation avec une distribution fixe de vaisseaux : la générer une seule fois aléatoirement, et la réutiliser pour l'instant

Idées d'amélioration : 
	- est-il possible de paralléliser certains calculs, par exemple avec les extensions AVX ?
		- peu probable, les coefficients des systèmes sont très inter-dépendants
	- 
