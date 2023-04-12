# First-repo
Ceci est le premier projet en python que je poste sur GitHub, tout feedback est apprécié!

Simulation du mouvement d'un pendule simple avec frottement, Raphael Pinson, Avril 2023

On utilise scipy et matplotlib pour animer le mouvement d'un pendule simple, 2D et 3D
Les fichiers Simplepend2D.py et Simplepend3D.py s'utilisent depuis le main.py
Toutes les fonctions ont un docstring accessible avec help(f) ou f.__doc__
Les simulations 2D et 3D s'utilisent avec la même syntaxe:

0/Importer Simplepend2D et/ou Simplepend3D selon vos besoins
1/Appeler set_constants(**parameters) pour choisir les paramètres de la simulation (détail + bas)
2/Appeler generate_states() pour effectuer la simulation
3/Pour générer le gif de la trajectoire, appeler animate_positions() 
4/Pour générer le gif de la trajectoire dans l'espace des phases, appeler animate_phasespace() 
5/Pour voir l'énergie potentielle/cinétique, appeler plot_energies()

Deux fonctions supplémentaires sont aussi disponibles dans la simulation 2D
-get_period()
Trace le graphe de theta(t) et compare avec la période théorique
retourne la période expérimentale et la période théorique du pendule (2 floats)

-small_angle_errors(N:int,thetazero:float,thetamax:float)
Effectue N simulations avec des angles de départ allant de thetazero à thetamax
Ensuite, compare pour chaque simulation période théorique et période réelle
Trace ensuite le graphe avec erreur absolue et erreur relative

Gestion des paramètres de la simulation:

Constantes de la simulation:
g: gravité (fixé a 9.81 m/s²)
l: longueur du fil en m
m: masse du pendule en kg
k: constante de frottement (unité SI)

Options du programme:
Debugmode: False par défaut, si True les fonctions affichent des informations dans la console
Showplots: True par défaut, contrôle si on affiche les graphiques à l'écran
Saveplots: False par défaut, contrôle si on sauvegarde les graphiques sur le disque
option: Contrôle la méthode d'intégration utilisée, valeurs valides: 'Euler','RK2','RK4'

Toutes ces variables sont modifiables depuis le main en appelant set_constants(**kwargs)
exemple d'utilisation: set_constants(k=0.2,m=1.6,Saveplots = True)
Le main.py est déja équipé d'un dictionnaire de paramètres et d'une instruction set_constants(**parameters)
