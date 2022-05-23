phi_v = 0.10 # densité en vaisseaux sanguins dans la grille
kN = 1e-4 # s^(-1) : constante de consommation du glucose par cellules normales (p.7 patel)
kT = 1e-3 # s^(-1) : constante de consommation du glucose par cellules cancéreuses
DG = 9.1e-5 # cm2/s
DH = 1.08e-5 #
delta = 2e-3 # 20 µm i.e 2e-3 cm 
q_G = 3.0e-5 # perméabilité de la paroi du vaisseau (cm/s) au glucose
q_H = 1.19e-4 # pareil, mais aux ions H+ (cm/s)

pH_D_N, pH_D_T = 6.8, 6.0 # pH seuils qui causent la mort d'une cellule (normale (N), resp. tumeur (T))
pH_Q_N, pH_Q_T = 7.1, 6.4 # pH seuils pour qu'une cellule soit dans un état "quiescent" (calme)

H_A, H_Q = 3e-5, 5e-7 # taux de production d'acide des cellules cancéreuses, en mM/s (milimoles par litre par secondes)

G_D = 2.5 # 2.5 mM, niveau de glucose minimum pour assurer la survie d'une cellule (normale ou cancéreuse confondues)
G_S = 5.0 # mM, niveau par défaut de la répartition de glucose (mM = milimole par litre; autre unité courament utilisée : miligram par décilitre (cf internet pour conversion))
H_S = 3.98e-5 # concentration en ions H+ du sérum/sang (i.e pH = 7.4)

