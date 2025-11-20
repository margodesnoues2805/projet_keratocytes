import numpy as np
from numba import jit

#---------------------INITIALISATION DE LA CHAINE---------------------

"""
On initialise une chaine de N maillons equirepartis sur un cercle de rayon R.

    Les arguments sont : 
            N : nombre de maillons (int)
            R : le rayon du cercle initial (float) en microns
            
    La fonction renvoie : 
            x et y : coordonnees xi et yi de chaque maillons Mi a t=0 (np.ndarray) 
            etat : etats (+1 := protusion ou -1 := rectraction) de chaque maillon Mi a t=0 (np.ndarray) 
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
            x : abcisses x[i] de chaque maillons Mi en t (np.ndarray) 
            y : ordonnees y[i] de chaque maillons Mi en t (np.ndarray) 
 
    La fonction renvoie : 
            aire : aire du polygone fermé (float)
            xG et yG : coordonnees xG et yG du barycentre G (float)
"""
@jit     
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
        p_vectoriel = (x[i] * y[j]) - (x[j] * y[i])
        aire += p_vectoriel
        somme_xG += (x[i] + x[j]) * p_vectoriel
        somme_yG += (y[i] + y[j]) * p_vectoriel

    # Calcul de l'aire totale de la cellule
    aire = aire * 0.5

    # Calcul du barycentre de la cellule
    xG = somme_xG / (6 * aire)
    yG = somme_yG / (6 * aire)

    return aire, xG, yG


#----------------------DEPLACEMENT DES MAILLONS-----------------------

"""                  
On calcul le deplacement entre t et t+1 pour chaque maillon Mi selon son etat,
puis on calcule les nouvelles coordonnees de chaque maillon.
        
        Les arguments sont :
            x : abcisses x[i] de chaque maillons Mi en t (np.ndarray) 
            y : ordonnees y[i] de chaque maillons Mi en t (np.ndarray)
            etat : etats (+1 := protusion ou -1 := rectraction) de chaque maillon Mi en t (np.ndarray)
            v_plus : vitesse des maillons en protusion (float)
            v_moins : vitesse des maillons en retraction (proportionnel a la distance entre Mi et G) 
            mode : mode de deplacement des maillons en protusion ('perpendiculaire' ou 'centrifuge')
            
        La fonction renvoie : 
            x_t1 : abcisses x[i] de chaque maillons Mi en t+1 (np.ndarray) 
            y_t1 : ordonnees y[i] de chaque maillons Mi en t+1 (np.ndarray)
"""

@jit
def deplacement_maillons_vector(x, y, etat, v_plus, v_moins, mode='perpendiculaire'):

    # Initialisation des variables
    N = len(x)
    x_t1 = np.copy(x)
    y_t1 = np.copy(y)

    # Calcul du barycentre
    aire, xG, yG = Calcul_Aire_Barycentre(x, y)

    # Masque des protrusions et rétractions
    masque_plus = etat == 1
    masque_moins = etat == -1



    # -------Protusion-------

    if mode == 'perpendiculaire':
        indices_precedent = np.roll(np.arange(N), 1)
        indices_suivant = np.roll(np.arange(N), -1)
        dx = x[indices_suivant] - x[indices_precedent]
        dy = y[indices_suivant] - y[indices_precedent]
        norme = np.sqrt((-dy)**2 + (dx)**2)
        vecteur_x_plus = v_plus * ((-dy) / norme)
        vecteur_y_plus = v_plus * ((dx) / norme)
        #vecteur (-dy,dx) normalise * norme de vitesse

    elif mode == 'centrifuge':
        dx = x - xG
        dy = y - yG
        norme = np.sqrt(dx**2 + dy**2)
        #nouveau tableau pour avoir les distances 
        vecteur_x_plus = np.zeros_like(dx)
        vecteur_y_plus = np.zeros_like(dy)
        #masque pour eviter division par zero si jamais un point est proche du barycentre
        masque_nonzero = norme > 0
        # vecteurs de deplacement
        vecteur_x_plus[masque_nonzero] = v_plus * dx[masque_nonzero] / norme[masque_nonzero]
        vecteur_y_plus[masque_nonzero] = v_plus * dy[masque_nonzero] / norme[masque_nonzero]

    else:
        raise ValueError("Donner un mode de deplacement parmis : 'perpendiculaire' ou 'centrifuge'")

    # On applique le deplacement aux protrusions
    x_t1[masque_plus] += vecteur_x_plus[masque_plus]
    y_t1[masque_plus] += vecteur_y_plus[masque_plus]



    # -------Retraction-------
    
    dx_G = x - xG
    dy_G = y - yG
    distance = np.sqrt(dx_G**2 + dy_G**2)
    #nouveau tableau pour avoir les distances
    vecteur_x_moins = np.zeros_like(dx_G)
    vecteur_y_moins = np.zeros_like(dy_G)
    #masque pour eviter division par zero si jamais un point est proche du barycentre
    nonzero = distance > 0
    # vecteurs de deplacement
    vecteur_x_moins[nonzero] = -v_moins(distance[nonzero]) * dx_G[nonzero] / distance[nonzero]
    vecteur_y_moins[nonzero] = -v_moins(distance[nonzero]) * dy_G[nonzero] / distance[nonzero]

    # On applique le deplacement aux retractions
    x_t1[masque_moins] += vecteur_x_moins
    y_t1[masque_moins] += vecteur_y_moins

    return x_t1, y_t1