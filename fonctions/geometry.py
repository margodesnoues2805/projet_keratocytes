import numpy as np
from numba import jit
from utils import vitesse_moins, signe_angle, chgmt_etat_plus, chgmt_etat_moins

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

# -------------------- INITIALISATION DE LA FORME DE LA CELLULE ---------------

def forme_initiale(N, forme="cercle", R=1.0, epsilon=0.2):
    """
    Génère une forme initiale :
    - cercle
    - etoile
    - polygone_complexe
    - asymetrie (portion en protusion, portion en rétraction)
    - polarisee (goutte / kératocyte)
    - coeur
    """
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    # ------------------------------- 1) CERCLE -------------------------------
    if forme == "cercle":
        r = R * np.ones(N)


    # ------------------------------- 2) ÉTOILE -------------------------------
    elif forme == "etoile":
        k = 5
        r = R * (1 + epsilon * np.cos(k * angles))


    # ------------------------------- 3) POLYGONE COMPLEXE --------------------
    elif forme == "polygone_complexe":
        bruit = epsilon * np.random.randn(N)
        r = R * (1 + bruit)
        r = 0.5*(r + np.roll(r, 1))   # lissage


    # --------------- 4) ASYMÉTRIQUE : moitié protusion, moitié rétraction ----
    elif forme == "asymetrie":
        r = R * np.ones(N)

        # Portion avant (protusion)
        avant = (angles < np.pi)  # demi-cercle
      
        # Portion arrière (rétraction)
        arriere = (angles >= np.pi)

        r[avant] += epsilon * R        # protusion
        r[arriere] -= epsilon * R      # rétraction
        
        # sécurité : pas de rayon négatif
        r = np.maximum(r, 0.1*R)


    # --------------------------- 5) FORME POLARISÉE --------------------------
    elif forme == "polarisee":
        # Forme goutte d'eau (kératocyte polarisé)
        r = R * (1 + epsilon * np.cos(angles))  # protusion devant
        
        # Avant bien marqué (protrusion)
        # arctan = adoucit la transition
        r *= (1 + 0.5 * np.tanh(3 * np.cos(angles)))


    # --------------------------- 6) FORME DE COEUR ---------------------------
    elif forme == "coeur":
        # Équation polaire standard
        r = R * (1 - np.sin(angles))

        # On peut amplifier si tu veux un cœur plus "profond"
        r *= (1 + epsilon)


    # --------------------------- FORME NON RECONNUE --------------------------
    else:
        raise ValueError("Forme doit être : cercle / etoile / polygone_complexe / asymetrie / polarisee / coeur")


    # ---------------------------- COORDONNÉES FINALES ------------------------
    x = r * np.cos(angles)
    y = r * np.sin(angles)

    # États aléatoires pour commencer
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
        j = (i + 1) %N # modulo N 
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

def deplacement_maillons_vector(x, y, etat, v_plus, R_plus, R_moins, mode='perpendiculaire'):

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
        # Eviter division par zero
        norme[norme == 0] = 1.0
        vecteur_x_plus = v_plus * ((dy) / norme)
        vecteur_y_plus = v_plus * ((-dx) / norme)
        #vecteur (dy,-dx) normalise * norme de vitesse
        
        
    elif mode == 'centrifuge':
        dx = x - xG
        dy = y - yG
        norme = np.sqrt(dx**2 + dy**2)
        #nouveau tableau pour avoir les distances 
        vecteur_x_plus = np.zeros_like(dx)
        vecteur_y_plus = np.zeros_like(dy)
        # vecteurs de deplacement
        vecteur_x_plus = v_plus * dx / norme
        vecteur_y_plus = v_plus * dy / norme

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
    # tableau des vitesses
    v_moins = vitesse_moins(distance, v_plus, R_plus, R_moins)
    # vecteurs de deplacement
    vecteur_x_moins[nonzero] = - v_moins[nonzero] * dx_G[nonzero] / distance[nonzero]
    vecteur_y_moins[nonzero] = - v_moins[nonzero] * dy_G[nonzero] / distance[nonzero]

    # On applique le deplacement aux retractions
    x_t1[masque_moins] += vecteur_x_moins[masque_moins]
    y_t1[masque_moins] += vecteur_y_moins[masque_moins]

    return x_t1, y_t1, aire, xG, yG

#----------------------Changement des etats------------------------
"""
 On change l'etat de chaque maillon Mi apres deplacement selon les regles definies.

    Les arguments sont :
            x : abcisses x[i] de chaque maillons Mi en t+1 (np.ndarray) 
            y : ordonnees y[i] de chaque maillons Mi en t+1 (np.ndarray)
            etat : etats de chaque maillon Mi en t (np.ndarray)
            R_plus : rayon maximal de protusion (float)
            R_moins : rayon minimal de retraction (float)
            xG et yG : coordonnees xG et yG du barycentre G (float)
            V : nombre de plus proches voisins ppv (multiple de 2) (int)
 
    La fonction renvoie : 
            nv_etat : nouveaux etats de chaque maillon Mi en t+1 (np.ndarray)
"""

def changement_etat(x, y, etat, R_plus, R_moins, xG, yG, V):
    
    # Parametres
    N = len(etat)

    # Initialisation
    nv_etat = np.copy(etat)
    dx_G = x - xG
    dy_G = y - yG
    distance = np.sqrt(dx_G**2 + dy_G**2)
    
    for i in range(N): 
    
        # -------Protusion-------
        if etat[i]== 1 :
            nv_etat[i] = chgmt_etat_plus(i, etat, distance[i], R_plus, V)
        
         # -------Retraction-------
        if etat[i]== -1 :
             nv_etat[i] = chgmt_etat_moins(i, etat, distance[i], R_moins, V)
             
    return nv_etat

#---------------------Elimination des boucles----------------------

"""
On supprime les maillons Mi créant des boucle (angle négatif) puis les 
réinsère selon le mode choisi : de maniere aleatoire ('random') ou entre 
les maillons les plus eloignes ('distance').

        Les arguments sont :
            x : abcisses x[i] de chaque maillons Mi en t+1 (np.ndarray) 
            y : ordonnees y[i] de chaque maillons Mi en t+1 (np.ndarray)
            etat : etats (+1 := protusion ou -1 := rectraction) de chaque maillon Mi en t+1 (np.ndarray)
            mode_reinsertion : 'random' ou 'distance'

        La fonction renvoie : 
            x : abcisses sans boucles x[i] de chaque maillons Mi en t+1 (np.ndarray) 
            y : ordonnees sans boucles y[i] de chaque maillons Mi en t+1 (np.ndarray)
            etat : etats sans boucles de chaque maillon Mi en t+1 (np.ndarray)
            
""" 
    
def elimination_boucles(x, y, etat, xG, yG, R_plus, R_moins, V, mode_reinsertion="random"):
    
    # Parametres
    seuil = 1e-12  # seuil pour éviter les fausses boucles
    N = len(x)

    # Copie en liste
    x = list(x)
    y = list(y)
    etat = list(etat)

    # Copie des états initiaux 
    etat_orig = etat.copy()
    x_orig = x.copy()
    y_orig = y.copy()

    candidats = []  # indices des maillons à supprimer

    # Recherche des boucles
    i = 0
    while i < len(x):
        M1 = i
        M2 = (i + 1) % len(x)
        produit_vect = signe_angle(x[M1], y[M1], x[M2], y[M2], xG, yG)

        if produit_vect < -seuil:
            candidats.append(M2)
            x.pop(M2)
            y.pop(M2)
            etat.pop(M2)
        else:
            i += 1

    # Si pas de boucle
    if len(candidats) == 0:
        #print("Pas de boucles")
        return np.array(x), np.array(y), np.array(etat)
    

    # Nombre de maillons restants
    N_sans_boucles = len(x)
    etat_supprimes = [etat_orig[i] for i in candidats]

    # Réinsertion des maillons supprimes
    for k in range(len(candidats)):
        
        if mode_reinsertion == "random":
            # on reinsert au hasard
            indice_reinsertion = np.random.randint(0, N_sans_boucles)
            
        elif mode_reinsertion == "distance":
            # on cherche la plus grande distance entre deux maillons consécutifs
            distances = []
            for i in range(N_sans_boucles):
                j = (i + 1) % N_sans_boucles
                d = np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
                distances.append(d)
            indice_reinsertion = np.argmax(distances)
            
        else:
            raise ValueError("mode_reinsert doit être 'random' ou 'distance'")

        # on insert au milieu du segment (indice_reinsertion → indice_reinsertion+1)
        nv_indice = (indice_reinsertion + 1) % N_sans_boucles
        x_reinsertion = (x[indice_reinsertion] + x[nv_indice]) / 2
        y_reinsertion = (y[indice_reinsertion] + y[nv_indice]) / 2
        
        #Calcul de la distance
        dx_G = x_reinsertion - xG
        dy_G = y_reinsertion - yG
        distance = np.sqrt(dx_G**2 + dy_G**2)
        
        # Calcul nv etat
        if etat_supprimes[k] == 1:
            etat_reinsertion = chgmt_etat_plus(k, etat, distance, R_plus, V)
        else:
            etat_reinsertion = chgmt_etat_moins(k, etat, distance, R_moins, V)
            
        # Insertion
        x.insert(nv_indice, x_reinsertion)
        y.insert(nv_indice, y_reinsertion)
        etat.insert(nv_indice, etat_reinsertion)
        N_sans_boucles += 1
    
    # verification
    if N_sans_boucles != N :
        print("Erreur lors de l'elimination des boucles !")

    print("Boucles")
    return np.array(x), np.array(y), np.array(etat) 
