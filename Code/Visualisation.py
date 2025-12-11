import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter



#----------------------VIDEO DU DEPLACEMENT-----------------------

"""
 On crée une vidéo MP4 de l'évolution de la cellule dans un environement 2D infini.

    Les arguments sont :
            x_hist, y_hist : historique des coordonées des maillons Mi à chaque pas d'enregistrement* (list)
            etat_hist : historique des états de tous les maillons à chaque pas d'enregistrement* (list)
            bary_hist : historique des barycentres à chaque pas d'enregistrement* (list)
            fps : frames per seconds (int)
            buffer : marge supplémentaire autour des positions pour fixer un cadre de visualisation (int)
 
    La fonction enregistre et affiche la video mp4.
            
 * Un pas d'enregistrement =  500 pas de temps (1 ms) = 0.5 s
"""
    
def Create_video(x_hist, y_hist, etat_hist, bary_hist, fps=30, buffer=5, filename="cellule_1.mp4"):

   # Convertion en arrays pour les calculs
    x_hist_arr = np.array(x_hist)
    y_hist_arr = np.array(y_hist)
    bary_arr = np.array(bary_hist)

    # Limites pour avoir un cadre fixe p
    xmin = np.min(x_hist_arr) - buffer
    xmax = np.max(x_hist_arr) + buffer
    ymin = np.min(y_hist_arr) - buffer
    ymax = np.max(y_hist_arr) + buffer

    # Creation de la figure
    fig, ax = plt.subplots(figsize=(7,7))
    writer = FFMpegWriter(fps=fps)
    full_path = filename + ".mp4"
    
    print(f"Création de la vidéo : {filename}")

    # CREATION DE LA VIDEO
    with writer.saving(fig, full_path, dpi=200):
        for frame in range(len(x_hist)):
            ax.clear()
            
            # Maillons
            couleurs = ["g" if e==1 else "r" for e in etat_hist[frame]]
            ax.scatter(x_hist_arr[frame], y_hist_arr[frame], c=couleurs, s=10)
            ax.plot(list(x_hist_arr[frame]) + [x_hist_arr[frame][0]],list(y_hist_arr[frame]) + [y_hist_arr[frame][0]],"k-", linewidth=1)

            # Trajectoire du barycentre 
            ax.plot(bary_arr[:frame+1,0], bary_arr[:frame+1,1], 'b--', linewidth=1)
            ax.scatter(bary_arr[frame,0], bary_arr[frame,1], c='blue', s=20)

            # Cadre fixe
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')

            # Video
            ax.set_title(f"Déplacement de la cellule — {frame}/{len(x_hist)-1} secondes")
            writer.grab_frame()

    plt.close(fig)
    print("Vidéo obtenue avec succès")
    
    
    
#----------------------DISTRIBUTION DES SWITCHS-----------------------

"""
 On affiche la distribution du nombre de changements d'états (switchs) en fonction du temps.

    Les arguments sont :
            plus_vers_moins : liste du nombre de changement d'etats (+1 -> -1) par pas d'enregistrement (np.array)
            moins_vers_plus : liste du nombre de changement d'etats (-1 -> +1) par pas d'enregistrement (np.array)
            dt : durée entre deux pas d'enregistrements (en secondes) -> 500 ms = 0.5 s
 
    La fonction enregistre et affiche le graphique.
            
 * Un pas d'enregistrement =  500 pas de temps (1 ms) = 0.5 s
"""

def Switch_distribution(plus_vers_moins, moins_vers_plus, dt=0.5, filename = "figure"):
   
    # Data
    frames = np.arange(len(plus_vers_moins))
    time = frames * dt

    # Affichage des data
    plt.figure(figsize=(10,5))
    plt.plot(time, plus_vers_moins, '-o', label="P → R", linewidth=2)
    plt.plot(time, moins_vers_plus, '-o', label="R → P", linewidth=2)

    # Plot
    plt.title("Nombre de switchs en fonction du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Nombre de switchs")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename + "_distribution_switch.pdf")
    plt.show()
   
   
   
#----------------------HEATMAP DES SWITCHS-----------------------

"""
 On affiche une une heatmap indiquant si un maillon a changé d'état entre deux 
 pas d'enregistrements successifs. 
 
    Les arguments sont :
            etat_hist : historique des états de tous les maillons à chaque pas d'enregistrement* (list)
 
    La fonction enregistre et affiche le graphique.
            
 * Un pas d'enregistrement =  500 pas de temps (1 ms) = 0.5 s
"""

def Switch_heatmap(etat_hist, filename = "figure"):

    # Data
    N = len(etat_hist[0])
    T = len(etat_hist) - 1     

    # Matrice : N maillons × T instants
    heatmap = np.zeros((N, T))
    for t in range(1, len(etat_hist)):
        prev = etat_hist[t-1]
        now  = etat_hist[t]
        heatmap[:, t-1] = (prev != now).astype(int)

    # Plot
    plt.figure(figsize=(12,6))
    plt.imshow(heatmap, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Switch (1 = changement d'état)")
    plt.xlabel("Temps (frame)")
    plt.ylabel("Index maillon")
    plt.title("Heatmap des switchs de maillon au cours du temps")
    plt.savefig(filename + "heatmap_switch.pdf")
    plt.show()

