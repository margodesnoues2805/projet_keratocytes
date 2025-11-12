import numpy as np
import matplotlib.pyplot as plt

# %%------------------TEST INITIALISATION DE LA CHAINE-------------------

from geometry import Chaine_initiale

"Plot"
x, y, etat = Chaine_initiale(50, 10e-6)
couleurs = ['g' if e == 1 else 'r' for e in etat]
plt.scatter(x, y, c=couleurs, s=50)
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

# %%