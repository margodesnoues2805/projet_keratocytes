import numpy as np


#---------------------INITIALISATION DE LA CHAINE---------------------

""""On initialise une chaine de N maillons equirepartis sur un cercle de rayon R.
    Les arguments sont : 
            N : nombre de maillons (int)
            R : le rayon du cercle initial (float)
    La fonction renvoie : 
            x et y : les coordonnees xi et yi de chaque maillons Mi (np.ndarray) 
            etat : les etats (+1 := protusion ou -1 := rectraction) de chaque maillon Mi (np.ndarray) 
"""

def Chaine_initiale(N, R):
    
    # Maillons Mi equirepartis sur un cercle de rayon R
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Coordonnées des maillons Mi
    x = R * np.cos(angles)
    y = R * np.sin(angles)
    
    # États aléatoires des maillons Mi
    etat = np.random.choice([-1, 1], size=N)
    
    return x, y, etat

#---------------------INITIALISATION DE LA CHAINE---------------------