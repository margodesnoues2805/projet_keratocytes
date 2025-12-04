import numpy as np
from numba import jit



#----------------------Vitesse retraction-----------------------
"""
 On calcule la norme de la vitesse de retraction.

    Les arguments sont :
            distance : distance MiG de chaque maillons Mi en t (np.ndarray) 
            v_plus : norme de vitesse de protusion (float)
            R_plus : rayon maximal de protusion (float)
            R_moins : rayon minimal de retraction (float)
 
    La fonction renvoie : 
            v_moins : norme de vitesse de retaction de chaque maillon Mi en t (np.ndarray)
"""

def vitesse_moins(distance, v_plus, R_plus, R_moins):

    # Calcul 
    calcul_1 = -7 * distance**2 + 14 * R_plus * distance + R_plus**2 + 8 * R_moins * (R_moins - 2 * R_plus)
    calcul_2 = 4 * (R_plus - R_moins)**2

    # Vitesse moins
    v_moins = v_plus * calcul_1 / calcul_2

    # v_moins ne doit pas être négative = les Mi(-1) vont toujours vers G
    v_moins = np.maximum(v_moins, 0)

    return v_moins



#----------------------Chgmt etat Mi(+1)-----------------------
"""
 On change l'etat de chaque maillon Mi(+1) apres deplacement selon les
regles suivantes :
                     P = 1 si (MiG) > R+
                     P = n-/V si (MiG) < R+

    Les arguments sont :
            i : indice du maillon
            etat : etats de chaque maillon Mi en t (np.ndarray)
            distance : distances (MiG) en t+1 (np.ndarray)
            R_plus : rayon maximal de protusion (float)
            V : nombre de plus proches voisins ppv (multiple de 2) (int)
 
    La fonction renvoie : 
            nv_etat_Mi_plus : nouvel etat de Mi en t+1 (float)
"""

def chgmt_etat_plus(i, etat, distance, R_plus, V):
    
    # Parametres
    N = len(etat)
    demi_V = int(V // 2)
    # Plus proches voisins modulo N
    ppv = [(i + k) % N for k in range(-demi_V, demi_V+1) if k != 0]
    
    # Code
    if distance >= R_plus : # Cas (MiG) > R+ => P(+1 -> -1) = 1
        nv_etat_Mi_plus = -1 
        #print("flag1")
    else : # Cas (MiG) < R+ => P(+1 -> -1) = n-/V
        n_moins = sum(etat[j] == -1 for j in ppv)
        nv_etat_Mi_plus = -1 if np.random.rand() < (n_moins/V) else 1
        #print("flag2")
        
    return nv_etat_Mi_plus



#----------------------Chgmt etat Mi(-1)-----------------------
"""
 On change l'etat de chaque maillon Mi(-1) apres deplacement selon les
regles suivantes : 
                     P = 1 si (MiG) < R-
                     P = n+/V si (MiG) > R-

    Les arguments sont :
            i : indice du maillon
            etat : etats de chaque maillon Mi en t (np.ndarray)
            distance : distances (MiG) en t+1 (np.ndarray)
            R_moins : rayon minimal de retraction (float)
            V : nombre de plus proches voisins ppv (multiple de 2) (int)
 
    La fonction renvoie : 
            nv_etat_Mi_moins : nouvel etat de Mi en t+1 (float)
"""

def chgmt_etat_moins(i, etat, distance, R_moins, V):
    
    # Parametres
    N = len(etat)
    demi_V = int(V // 2)
    # Plus proches voisins modulo N
    ppv = [(i + k) % N for k in range(-demi_V, demi_V+1) if k != 0] 
    
    # Code
    if distance <= R_moins : # Cas (MiG) < R- => P(-1 -> +1) = 1
        nv_etat_Mi_moins = 1 
        #print("flag3")
    else : # Cas (MiG) > R- => P(-1 -> +1) = n+/V
        n_plus = sum(etat[j] == 1 for j in ppv)
        nv_etat_Mi_moins = 1 if np.random.rand() < (n_plus/V) else -1
        #print("flag4")
        
    return nv_etat_Mi_moins



#----------------------Produit vectoriel----------------------- 
"""
 On calcule le produit vectoriel entre 3 points.

    Les arguments sont :
            xM1 et yM1 : coordonnees xi et yi de Mi (float) 
            xM2 et yM2 : coordonnees xi+1 et yi+1 de Mi+1 (float)
            xG et yG : coordonnees du barycentre G (float)
 
    La fonction renvoie : 
            produit_vect : produit vectoriel (float)
"""

def signe_angle(xM1, yM1, xM2, yM2, xG, yG):
    produit_vect = ((xM1 - xG) * (yM2 - yG)) - ((yM1 - yG) * (xM2 - xG))
    return produit_vect







