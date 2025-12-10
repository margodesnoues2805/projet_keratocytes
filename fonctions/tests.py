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
    print("les longueurs des tableaux sont égales.")
else :
    print("Les longueurs ne sont pas égales, la fonction ne fonctionne pas.")
    

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

# Parametres
N = 30              # nombre de maillons
R = 1         # rayon initial
R_plus = 3     # rayon max de protusion
R_moins = 0.5     # rayon min de retraction
v_plus = 1.0        # vitesse des protrusions
n_steps = 1      # nombre de pas de temps


# Initialisation
x, y, etat = Chaine_initiale(N, R)

print("Positions initiales (x, y) :")
print(np.column_stack((x, y)))
print("État des maillons :")
print(etat)

# Stockage de l'historique des positions
x_hist = [x.copy()]
y_hist = [y.copy()]

# Boucle de deplacement 
for t in range(n_steps):
    x, y, aire, xG, yG = deplacement_maillons_vector(x, y, etat, v_plus, R_plus, R_moins, mode='centrifuge')
    x_hist.append(x.copy())
    y_hist.append(y.copy())

# Affichage deplacements
plt.figure(figsize=(7,7))
couleurs = ['g' if e == 1 else 'r' for e in etat]

plt.scatter(x_hist[0], y_hist[0], c=couleurs, s=60, label='t=0', edgecolor='k') # Positions initiales
plt.scatter(x_hist[-1], y_hist[-1], c=couleurs, s=60, label=f't={n_steps}', marker='x') # Positions finales

for i in range(N): # Lignes des déplacements
    xi_list = [x_hist[k][i] for k in range(n_steps+1)]
    yi_list = [y_hist[k][i] for k in range(n_steps+1)]
    plt.plot(xi_list, yi_list, 'gray', linestyle='--', linewidth=1)

plt.title(f"Déplacement des maillons sur {n_steps} pas")
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

# Verifications
print("\nPositions finales après", n_steps, "pas :")
print(np.column_stack((x, y)))

# %%----------------------TEST CHANGEMENT X PAS-----------------------

import matplotlib.pyplot as plt

from geometry import Chaine_initiale, deplacement_maillons_vector, changement_etat

# Parametres
V = 2
N = 30              # nombre de maillons
R = 1          # rayon initial
R_plus = 3      # rayon max de protusion
R_moins = 0.2     # rayon min de retraction
v_plus = 0.1       # vitesse des protrusions
n_steps = 3   # nombre de pas de temps


# Initialisation
x, y, etat = Chaine_initiale(N, R)

print("Positions initiales (x, y) et etat :")
print(np.column_stack((x, y, etat)))

# Stockage de l'historique des positions
x_hist = [x.copy()]
y_hist = [y.copy()]
etat_hist = [etat.copy()]
distance_hist = [1] * len(x)

# Boucle de deplacement 
for t in range(n_steps):
    x, y, aire, xG, yG = deplacement_maillons_vector(x, y, etat, v_plus, R_plus, R_moins, mode='centrifuge')
    x_hist.append(x.copy())
    y_hist.append(y.copy())
    dx_G_new = x - xG
    dy_G_new = y - yG
    distance_new = np.sqrt(dx_G_new**2 + dy_G_new**2)
    distance_hist.append(distance_new.copy())
    etat = changement_etat(x, y, etat, R_plus, R_moins, xG, yG, V)
    etat_hist.append(etat.copy())

# Affichage deplacements
# Couleurs selon l'état initial
couleurs_init = ['g' if e == 1 else 'r' for e in etat_hist[0]]

# Couleurs selon l'état final
couleurs_final = ['g' if e == 1 else 'r' for e in etat_hist[-1]]

# Affichage positions initiales
plt.scatter(x_hist[0], y_hist[0], 
            c=couleurs_init, s=60, label='t=0', edgecolor='k')

# Affichage positions finales
plt.scatter(x_hist[-1], y_hist[-1], 
            c=couleurs_final, s=60, label=f't={n_steps}', marker='x')

for i in range(N): # Lignes des déplacements
    xi_list = [x_hist[k][i] for k in range(n_steps+1)]
    yi_list = [y_hist[k][i] for k in range(n_steps+1)]
    plt.plot(xi_list, yi_list, 'gray', linestyle='--', linewidth=1)

plt.title(f"Déplacement des maillons sur {n_steps} pas")
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

# Verifications
print("\nPositions finales après", n_steps, "pas :")
print(np.column_stack((x, y, etat, distance_hist[-1])))

# %%----------------------TEST CHANGEMENT ETAT X PAS-----------------------

import matplotlib.pyplot as plt

from geometry import Chaine_initiale, deplacement_maillons_vector, changement_etat

"Parametres"

V = 2
N = 30              # nombre de maillons
R = 1          # rayon initial
R_plus = 3      # rayon max de protusion
R_moins = 0.1    # rayon min de retraction
v_plus = 0.1       # vitesse des protrusions
n_steps = 1   # nombre de pas de temps

" Initialisation "

x, y, etat = Chaine_initiale(N, R)
print("Positions initiales (x, y) et etat :")
print(np.column_stack((x, y, etat)))

" Stockage de l'historique des positions "

x_hist = [x.copy()]
y_hist = [y.copy()]
etat_hist = [etat.copy()]
distance_hist = [1] * len(x)

" Boucle de deplacement "

for t in range(n_steps):
    #deplacement
    x, y, aire, xG, yG = deplacement_maillons_vector(x, y, etat, v_plus, R_plus, R_moins, mode='centrifuge')
    x_hist.append(x.copy())
    y_hist.append(y.copy())
    # dstance
    dx_G_new = x - xG
    dy_G_new = y - yG
    distance_new = np.sqrt(dx_G_new**2 + dy_G_new**2)
    distance_hist.append(distance_new.copy())
    #changement etat
    etat = changement_etat(x, y, etat, R_plus, R_moins, xG, yG, V)
    etat_hist.append(etat.copy())

" Affichage deplacements "

# Couleurs selon l'état initial ou final
couleurs_init = ['g' if e == 1 else 'r' for e in etat_hist[0]]
couleurs_final = ['g' if e == 1 else 'r' for e in etat_hist[-1]]

# Affichage positions initiales et finales
plt.scatter(x_hist[0], y_hist[0], c=couleurs_init, s=60, label='t=0', edgecolor='k')
plt.scatter(x_hist[-1], y_hist[-1], c=couleurs_final, s=60, label=f't={n_steps}', marker='x')

# Lignes des déplacements
for i in range(N): 
    xi_list = [x_hist[k][i] for k in range(n_steps+1)]
    yi_list = [y_hist[k][i] for k in range(n_steps+1)]
    plt.plot(xi_list, yi_list, 'gray', linestyle='--', linewidth=1)

plt.title(f"Déplacement des maillons sur {n_steps} pas")
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

" Verification "

print("\nPositions finales après", n_steps, "pas :")
print(np.column_stack((x, y, etat, distance_hist[-1])))

# %%----------------------TEST ELIMINATION BOUCLES X PAS-----------------------

import matplotlib.pyplot as plt

from geometry import Chaine_initiale, deplacement_maillons_vector, changement_etat, elimination_boucles

"Parametres"

V = 2
N = 30              # nombre de maillons
R = 1          # rayon initial
R_plus = 3      # rayon max de protusion
R_moins = 0.1    # rayon min de retraction
v_plus = 0.1      # vitesse des protrusions
n_steps = 100   # nombre de pas de temps


" Initialisation "

x, y, etat = Chaine_initiale(N, R)
print("Positions initiales (x, y) et etat :")
print(np.column_stack((x, y, etat)))

" Stockage de l'historique des positions "

x_hist = [x.copy()]
y_hist = [y.copy()]
etat_hist = [etat.copy()]

" Boucle de deplacement "

for t in range(n_steps):
    
    # Déplacement
    x, y, aire, xG, yG = deplacement_maillons_vector(x, y, etat, v_plus, R_plus, R_moins, mode='centrifuge')

    # Changement d'état
    etat = changement_etat(x, y, etat, R_plus, R_moins, xG, yG, V)

    #  ELIMINATION DES BOUCLES AVANT DE CONTINUER 
    x, y, etat = elimination_boucles(x, y, etat, xG, yG, R_plus, R_moins, V, mode_reinsertion="random")

    # Sauvegarde des positions et etat apres elimination boucles
    x_hist.append(x.copy())
    y_hist.append(y.copy())
    etat_hist.append(etat.copy())

" Affichage deplacements "
# Couleurs selon l'état initial ou final
couleurs_init = ['g' if e == 1 else 'r' for e in etat_hist[0]]
couleurs_final = ['g' if e == 1 else 'r' for e in etat_hist[-1]]

# Affichage positions initiales et finales
plt.scatter(x_hist[0], y_hist[0], c=couleurs_init, s=30, label='t=0', edgecolor='k')
plt.scatter(x_hist[-1], y_hist[-1], c=couleurs_final, s=60, label=f't={n_steps}', marker='x')

# Relier les maillons finaux par un trait noir
plt.plot(x_hist[-1].tolist() + [x_hist[-1][0]], y_hist[-1].tolist() + [y_hist[-1][0]], 'k-', linewidth=1, label='chaîne finale')

# Lignes des déplacements
for i in range(N): 
    xi_list = [x_hist[k][i] for k in range(n_steps+1)]
    yi_list = [y_hist[k][i] for k in range(n_steps+1)]
    plt.plot(xi_list, yi_list, 'gray', linestyle='--', linewidth=1)

plt.title(f"Déplacement des maillons sur {n_steps} pas")
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

" Verification "

print("\nPositions finales après", n_steps, "pas :")
print(np.column_stack((x, y, etat)))