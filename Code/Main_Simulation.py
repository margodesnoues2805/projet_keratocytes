from Geometry import Simulation_cellule
from Outils import Compteur_switchs
from Visualisation import Switch_distribution, Switch_heatmap

"""
Choisissez l’emplacement où vous souhaitez enregistrer vos données :
"""

path = "C:/Users/Labtop/projet_keratocytes/Code/"  # emplacement d'enregistrement
name = "simulation_cercle_centri_1"            # nom de la simulation

filename = path + name


"""
Ce fichier a pour but de faire tourner la boucle complete du modele de déplacement d'un keratocyte.
L'utilisateur doit choisir les parametres suivants (ceux prédéfinis ici correspondent aux données 
éxperimentales observées chez les keratocytes): 
"""


N =  4096                     # nombre de maillons Mi
R = 12.5                      # rayon initial de la cellule en μm
epsilon = 0.2                 # amplitude des variations du rayon (0.2 => rayon R varie de 20%)
degre_polarisation = 0.5      # pourcentage de maillons en protusion pour une cellule polarisée 
R_plus= 20                    # rayon maximal de protusion en μm
R_moins= 5                    # rayon minimal de retraction en μm
v_plus= 2.5e-4                # norme de vitesse de protusion en μm/ms
V= 4                          # nombre de plus proches voisins (multiple de 2)
n_steps= 20000                # nombre de pas de temps


"""
Choisissez une forme de cellule initiale parmis : 
                    - "cercle"
                    - "etoile"
                    - "polygone_complexe"
                    - "cercle polarisé"
                    - "cellule polarisé"
                    - "coeur"
"""

forme= "cercle" 


"""
Choisissez un mode de déplacement pour les maillons en protusion parmis :
                    - 'perpendiculaire' = perpendiculaires aux maillons voisins
                    - 'centrifuge' = centriguge par rapport au baricentre de la cellule
"""             

mode="centrifuge"


"""
Vous pouvez lancer la simulation. La vidéo ainsi que les données seront directement sauvegardées dans le file_path. 
"""

# SIMULATION :
    
x_hist, y_hist, etat_hist, bary_hist, aires_hist = Simulation_cellule(N= N, 
                                                                      R= R,
                                                                      epsilon= epsilon, 
                                                                      degre_polarisation= degre_polarisation,
                                                                      R_plus= R_plus, 
                                                                      R_moins= R_moins, 
                                                                      v_plus= v_plus,
                                                                      V= V, 
                                                                      n_steps= n_steps,
                                                                      forme= forme, 
                                                                      mode=mode, 
                                                                      filename= filename )

# GRAPHIQUES:
    
# compteur de switch
plus_vers_moins, moins_vers_plus = Compteur_switchs(etat_hist)

# Distribution des switchs par type
Switch_distribution(plus_vers_moins, moins_vers_plus, dt=0.5, filename = filename)

# Heatmap des switchs
Switch_heatmap(etat_hist, filename = filename)