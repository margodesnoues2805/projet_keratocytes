import numpy as np


#---------------------INITIALISATION DE LA CHAINE---------------------

"""
On initialise une chaine de N maillons equirepartis sur un cercle de rayon R.

    Les arguments sont : 
            N : nombre de maillons (int)
            R : le rayon du cercle initial (float)
            
    La fonction renvoie : 
            x et y : coordonnees xi et yi de chaque maillons Mi (np.ndarray) 
            etat : etats (+1 := protusion ou -1 := rectraction) de chaque maillon Mi (np.ndarray) 
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

#------------------CALCUL DE L'AIRE ET DU BARYCENTRE------------------

"""
 On calcule l'aire et la position du barycentre G du polygone fermé en le decoupant en triangles.
 Attention, on cherche le centre de gravite de la cellule et pas le barycentre geometrique. 

    Les arguments sont :
            x : abcisses x[i] de chaque maillons Mi (np.ndarray) 
            y : ordonnees y[i] de chaque maillons Mi (np.ndarray) 
 
    La fonction renvoie : 
            aire : aire du polygone fermé (float)
            xG et yG : coordonnees xG et yG du barycentre G (float)
"""
    
def Calcul_Aire_Barycentre(x, y):

    # Initialisation des variables
    N = len(x)
    aire = 0.0
    somme_xG = 0.0
    somme_yG = 0.0

    # Calcul de la somme des produits vectoriels
    for i in range(N):
        # on trouve le point suivant : i+1
        j = (i + 1) % N # modulo N 
        cross = (x[i] * y[j]) - (x[j] * y[i])
        aire += cross
        somme_xG += (x[i] + x[j]) * cross
        somme_yG += (y[i] + y[j]) * cross

    # Calcul de l'aire totale de la cellule
    aire = abs(aire) * 0.5

    # Calcul du barycentre de la cellule
    xG = somme_xG / (6 * aire)
    yG = somme_yG / (6 * aire)

    return aire, xG, yG


#------------------------CALCUL DU BARYCENTRE-------------------------



