import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
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

# ------ Comptabiliser le nombre de switchs de P=>R et R=>P -----------

def compter_switchs(etat_hist):
    """
    Compte le nombre de switches de protusion -> retraction (P->R)
    et de retraction -> protusion (R->P) à chaque frame enregistrée.

    Arguments :
        etat_hist : liste de np.ndarray des états des maillons à chaque frame
                    (comme celui renvoyé par simulate_cellule)

    Retour :
        switch_P_to_R : liste du nombre de P->R à chaque frame
        switch_R_to_P : liste du nombre de R->P à chaque frame
    """
    switch_P_to_R = []
    switch_R_to_P = []

    for t in range(1, len(etat_hist)):
        etat_prev = etat_hist[t-1]
        etat_now = etat_hist[t]

#(etat_prev == 1) : tableau true/false, si maillon protusion => true
#(etat_now == -1) : tableau true/false, si maillon rétraction => true

        P_to_R = np.sum((etat_prev == 1) & (etat_now == -1)) #double condition 
        R_to_P = np.sum((etat_prev == -1) & (etat_now == 1)) #double condition 
        switch_P_to_R.append(P_to_R)
        switch_R_to_P.append(R_to_P)

    return switch_P_to_R, switch_R_to_P


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

# --------------------- Vidéo --------------------------------

def create_video(x_hist, y_hist, etat_hist, filename="cellule_1.mp4", fps=30, buffer=5):
    """
    Crée une vidéo MP4 de l'évolution de la cellule avec cadre fixe.
    
    Arguments :
    - x_hist, y_hist : listes des positions de tous les maillons à chaque enregistrement (toutes les 500 ms)
    - etat_hist : liste des états de tous les maillons à chaque enregistrement
    - filename : nom du fichier vidéo à sauvegarder
    - fps : frames par seconde
    - buffer : marge supplémentaire autour des positions pour le cadre
    """
    # Convertir en arrays pour faciliter les calculs
    x_hist_arr = np.array(x_hist)
    y_hist_arr = np.array(y_hist)

    # Calculer les limites globales pour un cadre fixe
    xmin = np.min(x_hist_arr) - buffer
    xmax = np.max(x_hist_arr) + buffer
    ymin = np.min(y_hist_arr) - buffer
    ymax = np.max(y_hist_arr) + buffer

    # Calculer les barycentres pour la trajectoire
    bary = np.array([[np.mean(x_hist_arr[f]), np.mean(y_hist_arr[f])]
                     for f in range(len(x_hist))])

    fig, ax = plt.subplots(figsize=(7,7))
    writer = FFMpegWriter(fps=fps)

    print(f"Création de la vidéo : {filename}")

    with writer.saving(fig, filename, dpi=200):
        for frame in range(len(x_hist)):
            ax.clear()

            # Couleurs des maillons selon état
            couleurs = ["g" if e==1 else "r" for e in etat_hist[frame]]
            ax.scatter(x_hist_arr[frame], y_hist_arr[frame], c=couleurs, s=10)

            # Fermer la cellule
            ax.plot(
                list(x_hist_arr[frame]) + [x_hist_arr[frame][0]],
                list(y_hist_arr[frame]) + [y_hist_arr[frame][0]],
                "k-", linewidth=1
            )

            # Tracer la trajectoire du barycentre jusqu'à la frame actuelle
            ax.plot(bary[:frame+1,0], bary[:frame+1,1], 'b--', linewidth=1)
            ax.scatter(bary[frame,0], bary[frame,1], c='blue', s=20)

            # Cadre fixe
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')

            ax.set_title(f"Cellule — frame {frame}/{len(x_hist)-1}")

            writer.grab_frame()

    plt.close(fig)
    print("Vidéo obtenue avec succès")
    
    
# --------------------- Distribution plot switch  --------------------------------  

def plot_switch_distribution(switch_P_to_R, switch_R_to_P, dt=0.5):
    """
    Affiche un graphe de la distribution des switchs en fonction du temps.
    
    Arguments :
        switch_P_to_R : liste du nombre de P->R par frame
        switch_R_to_P : liste du nombre de R->P par frame
        dt : durée entre deux enregistrements (secondes) -> 500 ms = 0.5 s
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    frames = np.arange(len(switch_P_to_R))
    time = frames * dt

    plt.figure(figsize=(10,5))
    plt.plot(time, switch_P_to_R, '-o', label="P → R", linewidth=2)
    plt.plot(time, switch_R_to_P, '-o', label="R → P", linewidth=2)

    plt.title("Nombre de switchs en fonction du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Nombre de switchs")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
  # --------------------- Distribution plot switch  --------------------------------  

  
def plot_switch_heatmap(etat_hist):
    """
    Produit une heatmap où chaque case indique si un maillon
    a changé d'état entre deux frames successives.

    etat_hist : liste d'arrays d'états (valeurs ±1)
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    N = len(etat_hist[0])       # nombre de maillons
    T = len(etat_hist) - 1      # nombre d'intervalles de temps

    # Matrice switchs : N maillons × T instants
    heatmap = np.zeros((N, T))

    for t in range(1, len(etat_hist)):
        prev = etat_hist[t-1]
        now  = etat_hist[t]
        heatmap[:, t-1] = (prev != now).astype(int)

    plt.figure(figsize=(12,6))
    plt.imshow(heatmap, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Switch (1 = changement d'état)")
    plt.xlabel("Temps (frame)")
    plt.ylabel("Index maillon")
    plt.title("Heatmap des switchs de maillon au cours du temps")
    plt.show()
    
    
