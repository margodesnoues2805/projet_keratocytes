import numpy as np
import pandas as pd


#----------------------VITESSE DE RETRACTION-----------------------
"""
 On calcule la norme de la vitesse de rétraction selon les paramètres du modèle
 et la distance entre Mi et le barycentre.

    Les arguments sont :
            distance : distance MiG de chaque maillons Mi en t (np.ndarray) 
            v_plus : norme de vitesse de protusion (float)
            R_plus : rayon maximal de protusion (float)
            R_moins : rayon minimal de rtraction (float)
 
    La fonction renvoie : 
            v_moins : norme de vitesse de rétaction de chaque maillon Mi en t (np.ndarray)
"""

def Vitesse_moins(distance, v_plus, R_plus, R_moins):

    # Calculs du modèle 
    calcul_1 = -7 * distance**2 + 14 * R_plus * distance + R_plus**2 + 8 * R_moins * (R_moins - 2 * R_plus)
    calcul_2 = 4 * (R_plus - R_moins)**2
    v_moins = v_plus * calcul_1 / calcul_2

    # Sécurité : v_moins ne doit pas être négative (les Mi(-1) vont toujours vers G)
    v_moins = np.maximum(v_moins, 0)

    return v_moins



#----------------------CHANGEMENT ETAT MI(+1)-----------------------
"""
 On change l'état d'un maillon Mi(+1) après déplacement selon les
règles suivantes :
                     - P = 1 si (MiG) > R+
                     - P = n-/V si (MiG) < R+

    Les arguments sont :
            i : indice du maillon
            etat : états de chaque maillon Mi en t (np.ndarray)
            distance : distances (MiG) en t+1 (np.ndarray)
            R_plus : rayon maximal de protusion (float)
            V : nombre de plus proches voisins (multiple de 2) (int)
 
    La fonction renvoie : 
            nv_etat_Mi_plus : nouvel état de Mi en t+1 (float)
"""

def Chgmt_etat_plus(i, etat, distance, R_plus, V):
    
    # Paramètres
    N = len(etat)
    demi_V = int(V // 2)
    # Plus proches voisins modulo N
    ppv = [(i + k) % N for k in range(-demi_V, demi_V+1) if k != 0]
    
    # REGLES CHANGEMENT ETAT
    
        # Cas (MiG) > R+ => P(+1 -> -1) = 1
    if distance >= R_plus : 
        nv_etat_Mi_plus = -1 
        
        # Cas (MiG) < R+ => P(+1 -> -1) = n-/V
    else : 
        n_moins = sum(etat[j] == -1 for j in ppv)
        nv_etat_Mi_plus = -1 if np.random.rand() < (n_moins/V) else 1
        
    return nv_etat_Mi_plus



#----------------------CHANGEMENT ETAT MI(-1)-----------------------
"""
 On change l'état d'un maillon Mi(-1) après déplacement selon les
 règles suivantes : 
                     P = 1 si (MiG) < R-
                     P = n+/V si (MiG) > R-

    Les arguments sont :
            i : indice du maillon
            etat : états de chaque maillon Mi en t (np.ndarray)
            distance : distances (MiG) en t+1 (np.ndarray)
            R_moins : rayon minimal de rétraction (float)
            V : nombre de plus proches voisins (multiple de 2) (int)
 
    La fonction renvoie : 
            nv_etat_Mi_moins : nouvel état de Mi en t+1 (float)
"""

def Chgmt_etat_moins(i, etat, distance, R_moins, V):
    
    # Paramètres
    N = len(etat)
    demi_V = int(V // 2)
    ppv = [(i + k) % N for k in range(-demi_V, demi_V+1) if k != 0] # Plus proches voisins
    
    # REGLES CHANGEMENT ETAT
    
        # Cas (MiG) < R- => P(-1 -> +1) = 1
    if distance <= R_moins : 
        nv_etat_Mi_moins = 1 
        
        # Cas (MiG) > R- => P(-1 -> +1) = n+/V
    else : 
        n_plus = sum(etat[j] == 1 for j in ppv)
        nv_etat_Mi_moins = 1 if np.random.rand() < (n_plus/V) else -1
        
    return nv_etat_Mi_moins



#----------------------PRODUIT VECTORIEL----------------------- 
"""
 On calcule le produit vectoriel entre 3 points.

    Les arguments sont :
            xM1 et yM1 : coordonnees xi et yi de Mi (float) 
            xM2 et yM2 : coordonnees xi+1 et yi+1 de Mi+1 (float)
            xG et yG : coordonnees du barycentre G (float)
 
    La fonction renvoie : 
            produit_vect : produit vectoriel (float)
"""

def Signe_angle(xM1, yM1, xM2, yM2, xG, yG):
    produit_vect = ((xM1 - xG) * (yM2 - yG)) - ((yM1 - yG) * (xM2 - xG))
    return produit_vect



#----------------------COMPTEUR DE SWITCHS-----------------------

"""
 la fonction permet de comptabiliser le nombre de changements d'états (+1 -> -1) 
 et (-1 -> +1) par pas d'enregistrement*.

    Les arguments sont :
            etat_hist : historique des états de tous les maillons à chaque pas d'enregistrement* (list)
 
    La fonction renvoie :
            plus_vers_moins : liste du nombre de changement d'etats (+1 -> -1) par pas d'enregistrement (np.array)
            moins_vers_plus : liste du nombre de changement d'etats (-1 -> +1) par pas d'enregistrement (np.array)
            
 * Un pas d'enregistrement =  500 pas de temps (1 ms) = 0.5 s
 """

def Compteur_switchs(etat_hist):
    
    # Initialisation
    plus_vers_moins = []
    moins_vers_plus = []

    # Boucle compteur
    for t in range(1, len(etat_hist)):
        
        # Changements d'états
        etat_prev = etat_hist[t-1]
        etat_now = etat_hist[t]
        
        # Compteur
        plus_vers_moins.append(np.sum((etat_prev == 1) & (etat_now == -1)))
        moins_vers_plus.append(np.sum((etat_prev == -1) & (etat_now == 1)))

    return plus_vers_moins, moins_vers_plus


