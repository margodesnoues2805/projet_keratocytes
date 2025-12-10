#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 16:01:07 2025

@author: xuan_nguyen
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

from geometry import (
    deplacement_maillons_vector,
    changement_etat,
    elimination_boucles,
    Calcul_Aire_Barycentre, 
    forme_initiale
)

# ===============================================================
#   MAIN LOOP
# ===============================================================

"""
     N : nombre de maillons 
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
    N=50,
    forme="cercle",
    R=1.0,
    epsilon=0.2,
    R_plus=3.0,
    R_moins=0.5,
    v_plus=0.1,
    V=2,
    n_steps=40,
):
    """
    Simulation complète de la migration du kératocyte :
    - forme initiale choisie
    - boucle déplacement + changement d’état + élimination boucles
    - étude de stabilisation
    """

    # ---- Initialisation ----
    x, y, etat = forme_initiale(N, forme=forme, R=R, epsilon=epsilon)
    aire_initiale, xG, yG = Calcul_Aire_Barycentre(x, y)
    print(f"Aire initiale={aire_initiale}")

    timer = 0  #compteur interne en ms 

    x_hist = [x.copy()]
    y_hist = [y.copy()]
    etat_hist = [etat.copy()]
    aires_hist = [aire_initiale]
    bary_hist = [[xG, yG]]
    switch_P_to_R_hist = [] #liste pour compter le nombre de switchs de protusion à rétraction
    switch_R_to_P_hist = [] #liste pour compter le nombre de switchs de rétraction à protusion 

    # ---- Boucle temporelle ----
    """"
    Dessiner du mode de déplacement entre 'centrifuge' et 'perpendiculaire'
    """
    
    for t in range(n_steps):
        # Déplacement
        x, y, aire, xG, yG = deplacement_maillons_vector(
            x, y, etat, v_plus, R_plus, R_moins, mode="centrifuge"
        )

        # Changement d’état
        etat = changement_etat(x, y, etat, R_plus, R_moins, xG, yG, V)

        # Élimination boucles
        x, y, etat = elimination_boucles(x, y, etat, xG, yG, R_plus, R_moins, V)
        
        #TIMER 
        timer += 1 #avance de 1 ms 
        
        if timer % 500 == 0: #toutes les x ms => enregistrement x, y, etat  
            
         #compter les switchs entre l'enregistrement précédent et maintenant 
            etat_prev = etat_hist[-1]
            P_to_R = np.sum((etat_prev == 1) & (etat == -1)) #somme les maillons précédemment en protusion et mtn en rétraction
            R_to_P = np.sum((etat_prev == -1) & (etat == 1)) #somme les maillons précédemment en rétraction et mtn en protusion 
            
            # Stockage des données 
            x_hist.append(x.copy())
            y_hist.append(y.copy())
            etat_hist.append(etat.copy())
            switch_P_to_R_hist.append(P_to_R)
            switch_R_to_P_hist.append(R_to_P)
            aires_hist.append(aire)  # Stockage aire après élimination boucles
            bary_hist.append([[xG, yG]])
            
            print(f"Enregistrement à t = {timer*1e-3:.3f} s")
            
            
     # --- BARRE DE PROGRESSION ---
        if t % 50 == 0 or t == n_steps - 1:  # mise à jour régulière
            progress = (t + 1) / n_steps
            bar_len = 30
            filled = int(progress * bar_len)
            bar = "█" * filled + "-" * (bar_len - filled)
            print(f"\rProgression : [{bar}] {progress*100:5.1f}% ({t+1}/{n_steps})", end="")

    # ===============================================================
    #   FIN DE SIMULATION : ANALYSES
    # ===============================================================
    

    # Aire finale
    aire_finale, xG_final, yG_final = Calcul_Aire_Barycentre(x, y)

    # Moyenne des aires enregistrées
    moyenne_aire = np.mean(aires_hist)
    ecart_type_aire = np.std(aires_hist)

    print(f"\nAire finale = {aire_finale:.4f}")
    print(f"Moyenne des aires échantillonnées = {moyenne_aire:.4f}")
    print(f"Écart-type des aires = {ecart_type_aire:.4f}")
    
#   ===============================================================
    #   GRAPHIQUE FINAL
    # ===============================================================

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
    
    

    return x_hist, y_hist, etat_hist, aires_hist, switch_P_to_R_hist, switch_R_to_P_hist

# %%===============================================================
#   SIMULATION 
# ===============================================================
"""
L'utilisateur choisit la forme 
"""


x_hist, y_hist, etat_hist, aires_hist, switch_P_to_R_hist, switch_R_to_P_hist = simulate_cellule(
    N=4096,
    forme="cercle",
    R=12.5,
    epsilon=0.2,
    R_plus=20,
    R_moins=5,
    v_plus=2.5e-4,
    V=4,
    n_steps=200000
)

create_video(x_hist, y_hist, etat_hist, filename="cellule_cercle.mp4", fps=20)

# Graphe du nombre de switchs par type
plot_switch_distribution(switch_P_to_R_hist, switch_R_to_P_hist, dt=0.5)

# Heatmap des switchs par maillon
plot_switch_heatmap(etat_hist)