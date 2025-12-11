import numpy as np
from numba import jit
from Outils import Vitesse_moins, Signe_angle, Chgmt_etat_plus, Chgmt_etat_moins, Sauvegarde_simulation_csv
from Visualisation import Create_video



# -------------------- INITIALISATION DE LA FORME DE LA CELLULE ---------------
"""
 On initialise la forme initiale de la cellule ainsi que les états initiaux des
 N maillons équirepartis tels que : +1 = protusion et -1 = rétraction.

    Les arguments sont :
            N : Nombre de maillons Ni (int) 
            R : Rayon initial de la cellule en μm (float)
            epsilon : Amplitude des variations du rayon en μm (float)
            degre_polarisation : pourcentage de maillons en protusion chez les cellules polarisées (float)
            forme : Choix de la forme par l'utilisateur entre :
                            - "cercle"
                            - "etoile"
                            - "polygone_complexe"
                            - "cercle polarisé"
                            - "cellule polarisé"
                            - "coeur"
 
    La fonction renvoie : 
            x et y : coordonnées xi et yi de chaque maillons Mi a t=0 (np.ndarray) 
            etat : états (+1 := protusion ou -1 := réctraction) de chaque maillon Mi
                   a t=0 (np.ndarray) 
"""

def Forme_initiale(N, R=1.0, epsilon=0.2, degre_polarisation=0.5, forme="cercle"):
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    # -------CERCLE-------
    if forme == "cercle":
        r = R * np.ones(N)
        # Etats aléatoires
        etat = np.random.choice([-1, 1], size=N) 

    # -------ÉTOILE-------
    elif forme == "étoile":
        k = 5
        r = R * (1 + epsilon * np.cos(k * angles))
        # Etats aléatoires
        etat = np.random.choice([-1, 1], size=N)

    # -------POLYGONE COMPLEXE-------
    elif forme == "polygone complexe":
        bruit = epsilon * np.random.randn(N)
        r = R * (1 + bruit)
        r = 0.5*(r + np.roll(r, 1)) # lissage
        # Etats aléatoires
        etat = np.random.choice([-1, 1], size=N)

    # -------CERCLE POLARISÉ-------
    elif forme == "cercle polarisé": 
        r = R * np.ones(N)
        # Etats polarisé
        nb_protusions = int(degre_polarisation * N)
        etat = -np.ones(N, dtype=int)
        etat[:nb_protusions] = 1 

    # -------CELLULE POLARISÉE-------
    elif forme == "cellule polarisée":
        r = R * (1 + epsilon * np.cos(angles))
        r *= (1 + 0.5 * np.tanh(3 * np.cos(angles))) # lissage avec arctan
        # Etats polarisé
        nb_protusions = int(degre_polarisation * N)
        etat = -np.ones(N, dtype=int)
        etat[:nb_protusions] = 1 

    # -------COEUR------- 
    elif forme == "coeur":
        r = (R * (1 - np.sin(angles))) * (1 + epsilon)
        # Etats aléatoires
        etat = np.random.choice([-1, 1], size=N)
        
        
    # -------FORME NON RECONNUE------- 
    else:
        raise ValueError('Forme doit être choisie parmis : "cercle", "étoile", "polygone complexe", "cercle polarisé", "cellule polarisée", "coeur"')


    # COORDONNÉES FINALES
    x = r * np.cos(angles)
    y = r * np.sin(angles)

    return x, y, etat



#------------------CALCUL DE L'AIRE ET DU BARYCENTRE------------------
"""
 On calcule l'aire et la position du barycentre G du polygone fermé en le découpant
 en triangles. Attention, on cherche le centre de gravité de la cellule et pas
 le barycentre géometrique. 

    Les arguments sont :
            x : abcisses x[i] de chaque maillons Mi en t (np.ndarray) 
            y : ordonnées y[i] de chaque maillons Mi en t (np.ndarray) 
 
    La fonction renvoie : 
            aire : aire du polygone fermé (float)
            xG et yG : coordonnées xG et yG du barycentre G (float)
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
On calcul le déplacement entre t et t+1 pour chaque maillon Mi selon son état,
puis on calcule les nouvelles coordonnées de chaque maillon.
        
        Les arguments sont :
            x : abcisses x[i] de chaque maillons Mi en t (np.ndarray) 
            y : ordonnées y[i] de chaque maillons Mi en t (np.ndarray)
            etat : états de chaque maillon Mi en t (np.ndarray)
            v_plus : vitesse des maillons en protusion (float)
            mode : mode de déplacement des maillons en protusion parmis :
                
                            - 'perpendiculaire' = mouvement perpendiculaires 
                                                  aux maillons voisins
                            - 'centrifuge' = mouvement centriguge par rapport 
                                             au baricentre de la cellule
            
        La fonction renvoie : 
            x_t1 : abcisses x[i] de chaque maillons Mi en t+1 (np.ndarray) 
            y_t1 : ordonnées y[i] de chaque maillons Mi en t+1 (np.ndarray)
"""

def Deplacement_maillons_vecteurs(x, y, etat, v_plus, R_plus, R_moins, mode='perpendiculaire'):

    # Initialisation des variables
    N = len(x)
    x_t1 = np.copy(x)
    y_t1 = np.copy(y)

    # Calcul du barycentre
    aire, xG, yG = Calcul_Aire_Barycentre(x, y)

    # Masque des protrusions et rétractions
    masque_plus = etat == 1
    masque_moins = etat == -1

    # -------PROTUSION-------

        # PERPENDICULAIRE
    if mode == 'perpendiculaire':
        indices_precedent = np.roll(np.arange(N), 1)
        indices_suivant = np.roll(np.arange(N), -1)
        dx = x[indices_suivant] - x[indices_precedent]
        dy = y[indices_suivant] - y[indices_precedent]
        norme = np.sqrt((-dy)**2 + (dx)**2)
        
        # Nouveaux tableau pour avoir les distances 
        vecteur_x_plus = np.zeros_like(dx)
        vecteur_y_plus = np.zeros_like(dy)
        
        # Eviter division par zero
        norme[norme == 0] = 1.0
        
        # Vecteurs de déplacement
        vecteur_x_plus = v_plus * ((dy) / norme)
        vecteur_y_plus = v_plus * ((-dx) / norme)
        
        # CENTRIFUGE
    elif mode == 'centrifuge':
        dx = x - xG
        dy = y - yG
        norme = np.sqrt(dx**2 + dy**2)
        
        # Nouveau tableau pour avoir les distances 
        vecteur_x_plus = np.zeros_like(dx)
        vecteur_y_plus = np.zeros_like(dy)
        
        # Vecteurs de déplacement
        vecteur_x_plus = v_plus * dx / norme
        vecteur_y_plus = v_plus * dy / norme

        # NON RECONNU
    else:
        raise ValueError("Donner un mode de deplacement parmis : 'perpendiculaire' ou 'centrifuge'")

    # On applique le déplacement aux protrusions
    x_t1[masque_plus] += vecteur_x_plus[masque_plus]
    y_t1[masque_plus] += vecteur_y_plus[masque_plus]
    

    # -------RETRACTION-------
    
    dx_G = x - xG
    dy_G = y - yG
    distance = np.sqrt(dx_G**2 + dy_G**2)
    
    # Nouveaux tableau pour avoir les distances
    vecteur_x_moins = np.zeros_like(dx_G)
    vecteur_y_moins = np.zeros_like(dy_G)
    
    # Tableau des vitesses
    v_moins = Vitesse_moins(distance, v_plus, R_plus, R_moins)
    
    # Vecteurs de déplacement
    vecteur_x_moins = - v_moins * dx_G / distance
    vecteur_y_moins = - v_moins * dy_G / distance

    # On applique le déplacement aux rétractions
    x_t1[masque_moins] += vecteur_x_moins[masque_moins]
    y_t1[masque_moins] += vecteur_y_moins[masque_moins]

    return x_t1, y_t1, aire, xG, yG


#----------------------CHANGEMENT D'ETATS------------------------
"""
 On change l'état de chaque maillon Mi après déplacement selon les règles définies.

    Les arguments sont :
            x : abcisses x[i] de chaque maillons Mi en t+1 (np.ndarray) 
            y : ordonnées y[i] de chaque maillons Mi en t+1 (np.ndarray)
            etat : états de chaque maillon Mi en t (np.ndarray)
            R_plus : rayon maximal de protusion (float)
            R_moins : rayon minimal de rétraction (float)
            xG et yG : coordonnées xG et yG du barycentre G (float)
            V : nombre de plus proches voisins (multiple de 2) (int)
 
    La fonction renvoie : 
            nv_etat : nouveaux états de chaque maillon Mi en t+1 (np.ndarray)
"""

def Changement_etat(x, y, etat, R_plus, R_moins, xG, yG, V):
    
    # Paramètres
    N = len(etat)

    # Initialisation
    nv_etat = np.copy(etat)
    dx_G = x - xG
    dy_G = y - yG
    distance = np.sqrt(dx_G**2 + dy_G**2)
    
    for i in range(N): 
    
        # -------PROTUSION-------
        if etat[i]== 1 :
            nv_etat[i] = Chgmt_etat_plus(i, etat, distance[i], R_plus, V)
        
         # -------RETRACTION-------
        if etat[i]== -1 :
             nv_etat[i] = Chgmt_etat_moins(i, etat, distance[i], R_moins, V)
             
    return nv_etat


#---------------------ELIMINATION DES BOUCLES----------------------
"""
On supprime les maillons Mi créant des boucle (angle négatif), puis on les 
réinsère dans la chaine selon le mode choisi. : de maniere aleatoire ('random') ou entre 
les maillons les plus eloignes ('distance').

        Les arguments sont :
            x : abcisses x[i] de chaque maillons Mi en t+1 (np.ndarray) 
            y : ordonnées y[i] de chaque maillons Mi en t+1 (np.ndarray)
            etat : états de chaque maillon Mi en t+1 (np.ndarray)
            mode_reinsertion : Mode de réinsertion choisi par l'utilisateur parmis :
                
                                - 'random' = réinsertion de manière aléatoire
                                - 'distance' = réinsertion entre les maillons 
                                              les plus distants entre eux

        La fonction renvoie : 
            x : abcisses sans boucles x[i] de chaque maillons Mi en t+1 (np.ndarray) 
            y : ordonnées sans boucles y[i] de chaque maillons Mi en t+1 (np.ndarray)
            etat : états sans boucles de chaque maillon Mi en t+1 (np.ndarray)
""" 
    
def Elimination_boucles(x, y, etat, xG, yG, R_plus, R_moins, V, mode_reinsertion="random"):
    
    # Paramètres
    seuil = 1e-12  # seuil pour éviter les fausses boucles
    N = len(x)

    # Passage en liste
    x = list(x)
    y = list(y)
    etat = list(etat)

    # Copie des états initiaux 
    etat_orig = etat.copy()

    # Indices des maillons à supprimer
    candidats = []  

    # RECHERCHE DES BOUCLES
    i = 0
    while i < len(x):
        M1 = i
        M2 = (i + 1) % len(x)
        produit_vect = Signe_angle(x[M1], y[M1], x[M2], y[M2], xG, yG)

        if produit_vect < -seuil:
            candidats.append(M2)
            x.pop(M2)
            y.pop(M2)
            etat.pop(M2)
        else:
            i += 1

    # PAS DE BOUCLES
    if len(candidats) == 0:
        #print("Pas de boucles")
        return np.array(x), np.array(y), np.array(etat)
    
    
    # PRESENCE DE BOUCLES
    
    # Nombre de maillons restants
    N_sans_boucles = len(x)
    etat_supprimes = [etat_orig[i] for i in candidats]

    # Réinsertion des maillons supprimés
    for k in range(len(candidats)):
        
        # Random
        if mode_reinsertion == "random":
            indice_reinsertion = np.random.randint(0, N_sans_boucles)
            
        # Distance
        elif mode_reinsertion == "distance":
            distances = []
            for i in range(N_sans_boucles):
                j = (i + 1) % N_sans_boucles
                d = np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
                distances.append(d)
            indice_reinsertion = np.argmax(distances)
            
        # MODE INCONNU  
        else:
            raise ValueError("mode_reinsert doit être 'random' ou 'distance'")

        # on réinsert au milieu du segment (indice_reinsertion → indice_reinsertion+1)
        nv_indice = (indice_reinsertion + 1) % N_sans_boucles
        x_reinsertion = (x[indice_reinsertion] + x[nv_indice]) / 2
        y_reinsertion = (y[indice_reinsertion] + y[nv_indice]) / 2
        
        #Calcul de la distance
        dx_G = x_reinsertion - xG
        dy_G = y_reinsertion - yG
        distance = np.sqrt(dx_G**2 + dy_G**2)
        
        # Calcul nouveau état
        if etat_supprimes[k] == 1:
            etat_reinsertion = Chgmt_etat_plus(k, etat, distance, R_plus, V)
        else:
            etat_reinsertion = Chgmt_etat_moins(k, etat, distance, R_moins, V)
            
        # Réinsertion
        x.insert(nv_indice, x_reinsertion)
        y.insert(nv_indice, y_reinsertion)
        etat.insert(nv_indice, etat_reinsertion)
        N_sans_boucles += 1
    
    # Sécurité : la chaine conserve le bon nombre de maillons
    if N_sans_boucles != N :
        print("Erreur lors de l'elimination des boucles !")

    print("Boucles")
    return np.array(x), np.array(y), np.array(etat) 



#---------------------BOUCLE COMPLETE DU MODELE----------------------

"""
 Simulation complète de la migration du kératocyte :

1) Initialisation de la cellule : 
    - Choix de la forme et génération des coordonnées et des états des maillons Mi -> Forme_initiale()
    - Calcul de l'aire et du barycentre initial -> Calcul_Aire_Barycentre()
    - Création des listes pour stocker x, y, etat, aire, barycentre, nombre de changements d'états (switchs)

2) Boucle avec n_pas d'itérations :
    - Déplacement des maillons Mi(x,y,etat,t) à Mi(x',y',etat,t+1) -> Deplacement_maillons_vecteurs()
    - Changement d'état des maillons Mi(x',y',etat,t+1) à Mi(x',y',etat',t+1) -> Changement_etat()
    - Élimination des boucles -> Elimination_boucles()
    - Stockage des données toutes les 500 pas de temps (dt=1ms) : x_hist, y_hist, etat_hist, aire_hist, bary_hist, switch_hist

3) Barre de progression pour suivre l'avancée de la simulation
    
4) Fin de la simulation, sauvegarde des données

5) Affichage de la video du déplacement de la cellule
    

    Les arguments sont :
            N : Nombre de maillons Ni (int) 
            R : Rayon initial de la cellule en μm (float)
            epsilon : Amplitude des variations du rayon en μm (float)
            degre_polarisation : pourcentage de maillons en protusion pour une cellule polarisée (float)
            R_plus : rayon maximal de protusion (float)
            R_moins : rayon minimal de rétraction (float)
            v_plus : vitesse des maillons en protusion (float)
            V : nombre de plus proches voisins (multiple de 2) (int)
            n_steps : nombre de pas de temps = 1ms à effectuer (int)
            forme : Choix de la forme par l'utilisateur entre :
                            - "cercle"
                            - "etoile"
                            - "polygone_complexe"
                            - "cercle polarisé"
                            - "cellule polarisé"
                            - "coeur"
            mode : mode de déplacement des maillons en protusion parmis :
                            - 'perpendiculaire' = perpendiculaires aux maillons voisins
                            - 'centrifuge' = centriguge par rapport au baricentre de la cellule
            filename : Nom du fichier pour le télechargement des données et de la vidéo
            
    La fonction enregistre et retourne les données (x_hist, y_hist, etat_hist, aire_hist, bary_hist, switch_hist)
    et affiche et enregistre la vidéo.
"""

def Simulation_cellule(N= 50, R= 1.0, epsilon= 0.2, degre_polarisation=0.0, R_plus= 3.0, R_moins= 0.5, v_plus= 0.1, V= 2, n_steps= 40, forme= "cercle", mode="centrifuge", filename= "simulation1"):

    # 1) INITIALISATION DE LA CELLULE
    x, y, etat = Forme_initiale(N, forme=forme, R=R, epsilon=epsilon)
    aire_initiale, xG, yG = Calcul_Aire_Barycentre(x, y)
    print(f"Aire initiale={aire_initiale}")
 
    # Initialisation du compteur de boucles
    timer = 0  
   
    # Création de listes pour stocker x, y, etat, aire de la cellule, barycentre, switchs
    x_hist = [x.copy()]
    y_hist = [y.copy()]
    etat_hist = [etat.copy()]
    aire_hist = [aire_initiale]
    bary_hist = [[xG, yG]]
   

    # 2) BOUCLE AVEC n PAS D'ITERATIONS   
    for t in range(n_steps):
        
        # Déplacement
        x, y, aire, xG, yG = Deplacement_maillons_vecteurs(x, y, etat, v_plus, R_plus, R_moins, mode="centrifuge")

        # Changement d’état
        etat = Changement_etat(x, y, etat, R_plus, R_moins, xG, yG, V)

        # Élimination boucles
        x, y, etat = Elimination_boucles(x, y, etat, xG, yG, R_plus, R_moins, V)
        
        # Compteur
        timer += 1 
       
        # Stockage des données toutes les 500 ms
        if timer % 500 == 0:
       
            #Recalcul du barycentre et de l'aire après élimination boucles
            aire, xG, yG = Calcul_Aire_Barycentre(x, y)
 
            #Ajout des données aux listes
            x_hist.append(x.copy())
            y_hist.append(y.copy())
            etat_hist.append(etat.copy())
            aire_hist.append(aire)
            bary_hist.append([xG, yG])  
           
            print(f"Enregistrement à t = {timer*1e-3:.3f} s")
           
    # 3) BARRE DE PROGRESSION 
        if t % 50 == 0 or t == n_steps - 1:  # mise à jour régulière
            progress = (t + 1) / n_steps
            bar_len = 30
            filled = int(progress * bar_len)
            bar = "█" * filled + "-" * (bar_len - filled)
            print(f"\rProgression : [{bar}] {progress*100:5.1f}% ({t+1}/{n_steps})", end="")
           

    # 4) FIN DE LA SIMULATION 

    # Aire finale
    aire_finale, xG_final, yG_final = Calcul_Aire_Barycentre(x, y)
    print(f"\nAire finale = {aire_finale:.4f}")
    
    # Sauvegarde
    Sauvegarde_simulation_csv(x_hist, y_hist, etat_hist, bary_hist, aire_hist, filename=filename)
   
    # 5) AFFICHAGE DE LA VIDEO
   
    Create_video(x_hist, y_hist, etat_hist, bary_hist, fps=30, buffer=5, filename= filename + "_video")
   
   
    return x_hist, y_hist, etat_hist, bary_hist, aire_hist 