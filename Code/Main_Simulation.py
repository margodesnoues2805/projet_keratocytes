from Geometry import Simulation_cellule
from Outils import Compteur_switchs
from Visualisation import Switch_distribution, Switch_heatmap


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

"""
L'utilisateur choisit la forme entre :
    cercle, etoile, polygone_complexe, asymetrie, polarisee, coeur
"""

# SIMULATION :
x_hist, y_hist, etat_hist, bary_hist, aires_hist = Simulation_cellule(N= 4096, 
                                                                      R= 12.5,
                                                                      epsilon= 0.2, 
                                                                      degre_polarisation=0.0,
                                                                      R_plus= 20, 
                                                                      R_moins= 5, 
                                                                      v_plus= 2.5e-4,
                                                                      V= 4, 
                                                                      n_steps= 5000,
                                                                      forme= "cercle", 
                                                                      mode="centrifuge", 
                                                                      filename= "simulation_cercle_centri_1")

# GRAPHIQUES:
    
# compteur de switch
plus_vers_moins, moins_vers_plus = Compteur_switchs(etat_hist)

# Distribution des switchs par type
Switch_distribution(plus_vers_moins, moins_vers_plus, dt=0.5, filename = "figure.png")

# Heatmap des switchs
Switch_heatmap(etat_hist, filename = "figure")