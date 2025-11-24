import numpy as np
import matplotlib.pyplot as plt

# %%------------------TEST INITIALISATION DE LA CHAINE-------------------

from geometry import Chaine_initiale

"Plot"
x, y, etat = Chaine_initiale(100, 10)
couleurs = ['g' if e == 1 else 'r' for e in etat]
plt.scatter(x, y, c=couleurs, s=3)
plt.gca().set_aspect('equal')
plt.show()

"Longeur des tableaux"
if len(x)==len(y)==len(etat):
    print("les longeurs des tableaux sont egale.")
else :
    print("Les longeurs ne sont pas egales, la fonction ne fonctionne pas.")
    

# %%----------------------TEST AIRE ET BARYCENTRE-----------------------

from geometry import Calcul_Aire_Barycentre

"Initialisation de la chaine (aire = pi, barycentre = [0,0]) "
x, y, etat = Chaine_initiale(50, 1)

"Test de la fonction"
aire_test, xG_test, yG_test = Calcul_Aire_Barycentre(x, y)
print(f"L'aire est egale a : {aire_test:.3f}") 
print(f"Le barycentre est en : G = [{xG_test:.2f}, {yG_test:.2f}]")


# %%----------------------TEST DEPLACEMENT X PAS-----------------------

import matplotlib.pyplot as plt

from geometry import Chaine_initiale, deplacement_maillons_vector

# -----------------------------
# PARAMÈTRES DU TEST
# -----------------------------
N = 30              # nombre de maillons
R = 10.0
R_plus = 1.0
R_moins = 0.25
v_plus = 1.0        # vitesse des protrusions
n_steps = 1        # nombre de pas de temps

# -----------------------------
# 1) Initialisation
# -----------------------------
x, y, etat = Chaine_initiale(N, R)

print("Positions initiales (x, y) :")
print(np.column_stack((x, y)))
print("État des maillons :")
print(etat)

# Stocker l'historique pour affichage des déplacements
x_hist = [x.copy()]
y_hist = [y.copy()]

# -----------------------------
# 2) Boucle de déplacement
# -----------------------------
for t in range(n_steps):
    x, y = deplacement_maillons_vector(x, y, etat, v_plus, R_plus, R_moins, mode='centrifuge')
    x_hist.append(x.copy())
    y_hist.append(y.copy())

# -----------------------------
# 3) Affichage avant/après
# -----------------------------
plt.figure(figsize=(7,7))
couleurs = ['g' if e == 1 else 'r' for e in etat]

# Positions initiales
plt.scatter(x_hist[0], y_hist[0], c=couleurs, s=60, label='t=0', edgecolor='k')

# Positions finales
plt.scatter(x_hist[-1], y_hist[-1], c=couleurs, s=60, label=f't={n_steps}', marker='x')

# Lignes pour visualiser tous les déplacements
for i in range(N):
    xi_list = [x_hist[k][i] for k in range(n_steps+1)]
    yi_list = [y_hist[k][i] for k in range(n_steps+1)]
    plt.plot(xi_list, yi_list, 'gray', linestyle='--', linewidth=1)

plt.title(f"Déplacement des maillons sur {n_steps} pas")
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

# -----------------------------
# 4) Vérification des tableaux finaux
# -----------------------------
print("\nPositions finales après", n_steps, "pas :")
print(np.column_stack((x, y)))