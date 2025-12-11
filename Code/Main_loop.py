import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

from Geometry import (
    Deplacement_maillons_vector,
    Changement_etat,
    Elimination_boucles,
    Calcul_Aire_Barycentre,
    Forme_initiale)

from Outils import Compteur_switchs

from Visualisation import (
    Create_video,
    Switch_distribution,
    Switch_heatmap)



"""
    N : nombre de maillons Mi
    forme : choix de la forme entre étoile, polygone complexe, et cercle
    R : rayon initial de la cellule
    epsilon : amplitude des variations du rayon (0.2 => rayon R varie de 20%)
    v_plus : norme de vitesse de protusion (float)
    R_plus : rayon maximal de protusion (float)
    R_moins : rayon minimal de retraction (float)
    V : nombre de plus proches voisins ppv (multiple de 2) (int)
    n_steps : nombre de pas de temps
"""

def simulate_cellule(
    N= 50,
    forme= "cercle",
    R= 1.0,
    epsilon= 0.2,
    R_plus= 3.0,
    R_moins= 0.5,
    v_plus= 0.1,
    V= 2,
    n_steps= 40,
):
    """
    Simulation complète de la migration du kératocyte
   
    1) Initialisation de la cellule
   
    a) Création de la cellule, forme et coordonnées x, y, état avec forme_initiale()
       Calcul de l'aire, xG, yG avec Calcul_Aire_Barycentre()
    b) Création de listes pour stocker x, y, etat, aire de la cellule, barycentre, switchs
   
   
    2) Boucle temporelle avec n_pas d'itérations :
   
    a) Maillon Mi x, y soumis à un déplacement avec deplacement_maillons_vector()
    b) Maillon Mi d'état -1 ou +1 soumis à un changement d'état (ou non) avec changement_etat()
    c) Élimination des boucles avec elimination_boucles()
    d) Stockage des données toutes les 500 ms
         Coordonnées des N maillons Mi : x, y, etat
         Aire de la cellule, coordonnées du barycentre xG, yG
         Nombre de switchs de Protusion à Rétraction & Rétraction à Protusion
   
       
       
    3) Barre de progression pour suivre l'avancée de la simulation
   
    4) Fin de la simulation, aire finale et aire moyenne restituées
   
    5) Graphique final
    """



    # 1) Initialisation de la cellule
   
    #1) a) Création de la cellule + Calcul de l'aire, xG, yG (coordonnées du barycentre)
   
    x, y, etat = forme_initiale(N, forme=forme, R=R, epsilon=epsilon)
    aire_initiale, xG, yG = Calcul_Aire_Barycentre(x, y)
    print(f"Aire initiale={aire_initiale}")

    timer = 0  #compteur interne en ms
   
    #1) b) Création de listes pour stocker x, y, etat, aire de la cellule, barycentre, switchs
    """

            - Coordonnées du maillon Mi : x, y (x_hist, y_hist)
            - État du maillon Mi entre Protusion (+1) et Rétraction (-1) (etat_hist)
            - Aire de la cellule (aires_hist)
            - Coordonnées du barycentre de la cellule (xG, yG)
            - Nombre de changements d'état de Protusion P à Rétraction R (switch_P_to_R)
            - Nombre de changements d'état de Protusion P à Rétraction R (switch_R_to_P)
       
    """

    x_hist = [x.copy()]
    y_hist = [y.copy()]
    etat_hist = [etat.copy()]
    aires_hist = [aire_initiale]
    bary_hist = [[xG, yG]]
    switch_P_to_R_hist = []
    switch_R_to_P_hist = []
   

    # 2) Boucle temporelle avec n_pas d'itérations :
       
    """" Choisir le mode de déplacement de la cellule entre 'centrifuge' et 'perpendiculaire'"""    
    for t in range(n_steps):
        # 2) a) Déplacement
        x, y, aire, xG, yG = deplacement_maillons_vector(
            x, y, etat, v_plus, R_plus, R_moins, mode="centrifuge"
        )

        # 2) b) Changement d’état
        etat = changement_etat(x, y, etat, R_plus, R_moins, xG, yG, V)

        # 2) c) Élimination boucles
        x, y, etat = elimination_boucles(x, y, etat, xG, yG, R_plus, R_moins, V)
       
        timer += 1 #Compteur interne en ms : avance de 1 ms
       
        # 2) d) Stockage des données toutes les 500 ms
        if timer % 500 == 0:
       
            #Recalcul du barycentre (xG, yG) pour tracer la trajectoire de la cellule
            aire, xG, yG = Calcul_Aire_Barycentre(x, y)
           
            #Comptage des switchs entre l'enregistrement précédent et maintenant
            etat_prev = etat_hist[-1] #état avant le changement d'état  
            P_to_R = np.sum((etat_prev == 1) & (etat == -1)) #somme les maillons précédemment en protusion et mtn en rétraction
            R_to_P = np.sum((etat_prev == -1) & (etat == 1)) #somme les maillons précédemment en rétraction et mtn en protusion
           
            #Ajout des données aux listes
            x_hist.append(x.copy())
            y_hist.append(y.copy())
            etat_hist.append(etat.copy())
            switch_P_to_R_hist.append(P_to_R)
            switch_R_to_P_hist.append(R_to_P)
            aires_hist.append(aire)  # Stockage aire après élimination boucles
            bary_hist.append([xG, yG])  
           
            print(f"Enregistrement à t = {timer*1e-3:.3f} s")
           
     # 3) Barre de progression de la simulation
        if t % 50 == 0 or t == n_steps - 1:  # mise à jour régulière
            progress = (t + 1) / n_steps
            bar_len = 30
            filled = int(progress * bar_len)
            bar = "█" * filled + "-" * (bar_len - filled)
            print(f"\rProgression : [{bar}] {progress*100:5.1f}% ({t+1}/{n_steps})", end="")
           

    #  4) Fin de la simulation, aire finale et aire moyenne restituées

    # Aire finale
    aire_finale, xG_final, yG_final = Calcul_Aire_Barycentre(x, y)

    # Moyenne des aires enregistrées
    moyenne_aire = np.mean(aires_hist)
    ecart_type_aire = np.std(aires_hist)

    print(f"\nAire finale = {aire_finale:.4f}")
    print(f"Moyenne des aires échantillonnées = {moyenne_aire:.4f}")
    print(f"Écart-type des aires = {ecart_type_aire:.4f}")
   
    # 5) Graphique final
   
    plt.figure(figsize=(7,7))
    couleurs_final = ['g' if e == 1 else 'r' for e in etat]

    plt.scatter(x_hist[0], y_hist[0], s=30, label="t=0", c='gray')
    plt.scatter(x_hist[-1], y_hist[-1], s=60, label=f"t={n_steps}", c=couleurs_final)

    plt.plot(list(x_hist[-1]) + [x_hist[-1][0]],
             list(y_hist[-1]) + [y_hist[-1][0]],
             'k-', linewidth=1)

    plt.gca().set_aspect("equal")
    plt.title("Évolution de la cellule")
    plt.legend()
    plt.show()
   
   

    return x_hist, y_hist, etat_hist, aires_hist, bary_hist, switch_P_to_R_hist, switch_R_to_P_hist

# %%===============================================================
#   SIMULATION
# ===============================================================
"""
L'utilisateur choisit la forme entre :
    cercle, etoile, polygone_complexe, asymetrie, polarisee, coeur
"""


x_hist, y_hist, etat_hist, aires_hist, bary_hist, switch_P_to_R_hist, switch_R_to_P_hist = simulate_cellule(
    N=4096,
    forme="etoile",
    R=12.5,
    epsilon=0.2,
    R_plus=20,
    R_moins=5,
    v_plus=2.5e-4,
    V=4,
    n_steps=5000
)

create_video(x_hist, y_hist, etat_hist, bary_hist, filename="cellule_etoile_2.mp4", fps=20)

# Graphe du nombre de switchs par type
plot_switch_distribution(switch_P_to_R_hist, switch_R_to_P_hist, dt=0.5)

# Heatmap des switchs par maillon
plot_switch_heatmap(etat_hist)