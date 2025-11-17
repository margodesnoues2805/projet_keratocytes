import numpy as np


#---------------------INITIALISATION DE LA CHAINE---------------------

"""
On initialise une chaine de N maillons equirepartis sur un cercle de rayon R.

    Les arguments sont : 
            N : nombre de maillons (int)
            R : le rayon du cercle initial (float) en microns
            
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
    aire = aire * 0.5

    # Calcul du barycentre de la cellule
    xG = somme_xG / (6 * aire)
    yG = somme_yG / (6 * aire)

    return aire, xG, yG


#----------------------DEPLACEMENT DES MAILLONS-----------------------

    """
    Déplace tous les maillons d'une étape de manière vectorisée.

    Paramètres
    ----------
    x, y : np.ndarray
        Coordonnées des maillons
    eta : np.ndarray
        États (+1 protrusion, -1 rétraction)
    v_plus : float
        Vitesse des maillons en protrusion
    v_moins : fonction
        Fonction v_moins(r) pour maillons en rétraction
    mode : str, 'perp' ou 'centrifuge'
        Méthode de déplacement pour les protrusions :
        - 'perp' : perpendiculaire à la chaîne
        - 'centrifuge' : radial depuis le barycentre

    Retourne
    -------
    x_t_1, y_t_1 : np.ndarray
        Nouvelles positions
    """

def deplacement_maillons_vector(x, y, etat, v_plus, v_moins, mode='perp'):

    # Initialisation des variables
    N = len(x)
    x_t1 = np.copy(x)
    y_t1 = np.copy(y)

    # Calcul du barycentre
    aire, xG, yG = calcul_aire_barycentre(x, y)

    # Masque des protrusions et rétractions
    masque_plus = etat == 1
    masque_moins = etat == -1

    # -----------------------------
    # Déplacement protrusions
    # -----------------------------
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
        vx_plus = np.zeros_like(dx)
        vy_plus = np.zeros_like(dy)
        #masque pour eviter division par zero
        masque_nonzero = norme > 0
        vx_plus[masque_nonzero] = v_plus * dx[masque_nonzero] / norme[masque_nonzero]
        vy_plus[masque_nonzero] = v_plus * dy[masque_nonzero] / norme[masque_nonzero]

    else:
        raise ValueError("mode doit être 'perp' ou 'centrifuge'")

    # Appliquer seulement aux protrusions
    x_t1[mask_plus] += vx_plus[mask_plus]
    y_t1[mask_plus] += vy_plus[mask_plus]

    # -----------------------------
    # Déplacement rétractions
    # -----------------------------
    dx_minus = x[mask_minus] - xG
    dy_minus = y[mask_minus] - yG
    r = np.sqrt(dx_minus**2 + dy_minus**2)

    vx_minus = np.zeros_like(dx_minus)
    vy_minus = np.zeros_like(dy_minus)
    nonzero = r > 0
    vx_minus[nonzero] = -v_moins(r[nonzero]) * dx_minus[nonzero] / r[nonzero]
    vy_minus[nonzero] = -v_moins(r[nonzero]) * dy_minus[nonzero] / r[nonzero]

    x_t1[mask_minus] += vx_minus
    y_t1[mask_minus] += vy_minus

    return x_t_1, y_t_1