import numpy as np
from numba import jit


def vitesse_moins(distance, v_plus, R_plus, R_moins):

    # Calcul :
    calcul_1 = -7 * distance**2 + 14 * R_plus * distance + \
        R_plus**2 + 8 * R_moins * (R_moins - 2 * R_plus)
    calcul_2 = 4 * (R_plus - R_moins)**2

    # Vitesse moins
    v_moins = v_plus * calcul_1 / calcul_2

    # v_moins ne doit pas être négative = les maillons en rétraction vont toujours vers G
    v_moins = np.maximum(v_moins, 0)

    return v_moins


def changement_etat(x, y, etat, R_plus, R_moins, aire, xG, yG, V):

    # -----Initialisation-----
    nv_etat = np.zeros_like(etat)
    dx_G = x - xG
    dy_G = y - yG
    distance = np.sqrt(dx_G**2 + dy_G**2)
    
    for i in len(etat): 
        
        # -------Protusion-------
        if etat[i]== 1 :
            
            if distance >= R_plus :
                nv_etat[i] = -1 
                
            else :
                n_moins = 0
                
                for j in (i-V/2, i+V/2):
                    if etat[j] == -1:
                        n_moins += 1
                
                nv_etat[i] == -1 if np.random.rand() < (n_moins/V) else 1
        
         # -------Retraction-------
        if etat[i]== -1 :
             
             if distance <= R_moins :
                 nv_etat[i] = 1 
                 
             else :
                 n_plus = 0
                 
                 for j in (i-V/2, i+V/2):
                     if etat[j] == 1:
                         n_plus += 1
                 
                 nv_etat[i] == 1 if np.random.rand() < (n_plus/V) else -1  
                 
    return nv_etat
