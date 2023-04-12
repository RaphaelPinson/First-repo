import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
from time import time, localtime, strftime
from inspect import isfunction,signature

Debugmode = False
Saveplots = False
Showplots = True
#Initialisation des constantes et conditions initiales
Nframes = 300
tmax = 30
framerate = 100
l = 2.0
g = 9.81
m = 0.7
k = 0.1
theta_0 = np.pi/6
phi_0 = np.pi/6
dtheta_0 = 0.0
dphi_0 = 0.0
option: str = "Euler"
options = ["Euler","RK2","RK4"]


#Nom des variables qu'on autorise à modifier depuis le main()
constants = ["Debugmode", "Saveplots","Showplots", "Nframes", "tmax", "framerate", "theta_0", "dtheta_0","phi_0","dphi_0", "l", "m","k","option"]
def set_constants(**kwargs):
    #Docstring de la fonction, s'affiche avec set_constants.__doc__
    """
    Arguments
    ----------
    **kwargs : Dictionnaire au format {nom : valeur,} pour chaque constante à modifier.
        Appeler print_variables() liste le nom des constantes modifiables.
        La fonction vous dit si le nom où le type d'une constante ne convient pas.
        
    
    Attention: la fonction utilise exec(nom = valeur) pour affecter automatiquement les valeurs aux constantes
    Plusieurs précautions ont été prises pour sécuriser les inputs mais faut pas faire nimp quand meme
    
    Exception
    ------
    ValueError
        Levée si exec(nom = valeur) renvoie une exception.
        Peut se produire avec des variables de type string et function
        Les constantes risquant de générer des bugs sont traités à part
        
    """
    
    start_time = time()
    #Boucle principale, typechecking et namechecking 
    for name in kwargs.keys():
        if name not in constants:
            print(f'Aucune variable nommée {name}')
        elif not isinstance(globals()[name],type(kwargs[name])):
            print(f'{name}doit être un {type(globals()[name])}')
        elif name != 'option': #option de calcul traitée à part
            if Debugmode:
                print('')
                print(f'{name} est de type {type(globals()[name])}')
                print(name + ' vaut par défaut: ' + str(globals()[name]))
                print(f'{name} doit être mis à {kwargs[name]}')
                print(f'{kwargs[name]} : {type(kwargs[name])}')
            try:
                #on va faire une fct à part pour gérer l'option car bug
                exec(f'{name} = {kwargs[name]}',globals()) #change une variable globale depuis l'interieur d'une fonction
                if Debugmode:
                    print(f'{name} vaut maintenant {globals()[name]}')
            except Exception:
                raise ValueError("impossible de mettre "+name+" à "+kwargs[name])
                
    #Cas à part: option est une variable de type string et fait bugger exec
    if 'option' in kwargs.keys():
        global option
        option = kwargs['option']
        while option not in options:
            option = input(f'saisir une méthode de calcul parmi {options}')
        
    if Debugmode:
        print(f'set_constants fini en {round(time() - start_time,6)} s')


def get_xyz(theta,phi):
    """
    Parameters
    ----------
    theta,phi: float: latitude/longitude 
    Returns
    -------
    [x,y,z]: float: vecteur position en coordonnées cartésiennes
    Note: z = -lcos(theta) car on veut z(theta = 0) = -l
    """
    return [l*np.sin(theta)*np.cos(phi),l*np.sin(theta)*np.sin(phi),-l*np.cos(theta)]

def get_dxyz(theta, dtheta, phi, dphi):
    """
    Arguments
    ----------
    theta, phi: float: latitude/longitude 
    dtheta, dphi: float: vitesses angulaires 

    Output
    -------
    [dx,dy,dz]: vecteur vitesse en coordonnées cartésiennes

    """
    dx: float = l*dtheta*np.cos(theta)*np.cos(phi) - l*dphi*np.sin(theta)*np.sin(phi)
    dy: float =  l*dtheta*np.cos(theta)*np.sin(phi) + l*dphi*np.sin(theta)*np.cos(phi)
    dz: float = l*dtheta*np.sin(theta)
    return [dx,dy,dz]


#fonction qui transforme S en dS/dt
#si on veut des valeurs + précises on peut faire un schéma de runge-kutta
#On suppose que la force de frottement marche pareil que pour le pendule 2D
def diff_S(S,t,g,l,k):
    """    
    Fonction qu'on rentre dans le schéma d'intégration
    
    Arguments
    ----------
    S = [theta,dtheta,phi,dphi] : etat du système à l'instant t
    t : float: temps
    g, l, k: float: constantes

    Output
    -------
    dS = [dtheta,ddtheta,dphi,ddphi]: dérivée de l'etat à l'instant t
    ddtheta et ddphi dépendent de l'equadiff propre au système

    """
    theta, dtheta, phi, dphi = S
    return[dtheta, np.sin(theta)*(-g/l + np.cos(theta)*dphi**2) - k*dtheta/m, dphi, -2*dphi*dtheta/np.tan(theta) -k*dphi/m]

def RungeKutta2(func, y0, t, args):
    """
    résout l'équation dy/dt = f(t,y(t),*args)
    
    Arguments
    ----------
    func : fonction qui transforme S en dS/dt
    y0 : état du système à t = 0 = (theta0,phi0)
    t : array contentant tous les instants 
    args : constantes de l'equation: (g,l,k) dans notre cas
    
    Output
    -------
    y : array des (theta(t),phi(t)) pour tout t
    
    """
    
    n= len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        y[i + 1] = y[i] + h * func(y[i] + (h/2.)*func(y[i], t[i], *args) , t[i] + h/2., *args)
    return y


def RungeKutta4(func, y0, t, args):
    """
    résout l'équation dy/dt = f(t,y(t),*args)
    
    Arguments
    ----------
    func : fonction qui transforme S en dS/dt
    y0 : état du système à t = 0 = (theta0,phi0)
    t : array contentant tous les instants 
    args : constantes de l'equation: (g,l,k) dans notre cas
    
    Output
    -------
    y : array des (theta(t),phi(t)) pour tout t
    """
    n= len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = func(y[i], t[i], *args)
        k2 = func(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = func(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = func(y[i] + k3 * h, t[i] + h, *args)
        y[i + 1] = y[i] + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

#on liste toutes les grandeurs qu'on étudie lors de la simulation
statenames = ["theta_vals","dtheta_vals","phi_vals","dphi_vals","xyzs","dxyzs","T_vals","V_vals","Em_vals","L_vals","times"]
States = {}
#fonction qui se charge de la simulation et note l'état du système pour tout t
def generate_states(minimal:bool):
    """
    Fonction principale du programme:
        
    Calcule l'etat du système pour tous les instants de times
    Utilise scipy.odeint, RK2 ou RK4 selon la valeur de 'option'
    Remplit le dictionnaire States (variable globale) une fois le calcul terminé
    Ce dictionnaire est ensuite utilisé par les fonctions d'animation et d'affichage
    
    Etats du système (tous des np.array de longueur Nframes):
    1/Etats indispensables (calculés dans tous les cas)
    -times: tableau des valeurs de t 
    -thetavals: tableau des valeurs de theta
    -dthetavals: tableau des valeurs de dtheta
    -phivals: tableau des valeurs de phi
    -dphivals: tableau des valeurs de dphi
    
    2/Etats calculés seulement si minimal est à False
    -xyzs: coordonnées cartésiennes au format [x,y,z] 
    -dxyzs: vecteurs vitesse au format [dx,dy,dz]
    -T_vals: valeurs de l'énergie cinétique (0.5*m*v^2)
    -V_vals: valeurs de l'énergie potentielle (m*g*(l+z))
    -L_vals: valeurs du lagrangien T - V
    -Em_vals: valeurs d'énergie mécanique T + V (hamiltonien)
    
    Arguments
    ----------
    minimal : bool
    Si mis à True, on ne calcule que les états indispensables (utile pour enchainer plusieurs simulations)
    Pour l'animation de la trajectoire, on doit donc mettre à False pour avoir les positions cartésiennes

    Output
    -------
    Ne retourne rien mais met à jour le dictionnaire States

    """
    start_time = time()
    
    #Choix de la méthode de calcul
    method = odeint
    if option =='RK2':
        method = RungeKutta2
    elif option == 'RK4':
        method = RungeKutta4
    
    if Debugmode:
        print("")
        print("Ce code simule un pendule simple 3D avec scipy")
        print(f'Méthode de calcul: {method.__name__}')
        print("masse du pendule: "+str(m))
        print("longueur du fil: " + str(l))
        print("constante de friction: "+str(k))
        print("etat initial:",end='\n')
        print("theta = "+str(round(theta_0,6))+" dtheta = "+str(round(dtheta_0,6)))
        print("phi = " + str(round(phi_0, 6)) + " dphi = " + str(round(dphi_0, 6)))
        print("")
    S_zero = [theta_0, dtheta_0, phi_0, dphi_0]
    global States
#Contient la liste des (theta,dtheta) pour tous t
    times = np.linspace(0, tmax, Nframes)
    phasestates = np.zeros((len(times), len(S_zero)))
    try:
        phasestates = method(diff_S, S_zero, times, args=(g, l, k))
    except Exception:
        raise NameError(f'{method.__name__} ne génère pas correctement les états')
#on extrait la liste des valeurs de theta et dtheta
    theta_vals = phasestates[:,0]
    dtheta_vals = phasestates[:,1]
    phi_vals =  phasestates[:,2]
    dphi_vals = phasestates[:,3]
    if minimal:
        States = {"theta_vals":theta_vals,
                  "dtheta_vals:":dtheta_vals,
                  "phi_vals":phi_vals,
                  "dphi_vals":dphi_vals,
                  "times":times}
        return None
#on récupère les positions cartésiennes pour l'animation
    xyzs  = np.array([get_xyz(theta,phi) for (theta,phi) in zip(theta_vals,phi_vals)])
#On calcule les énergies cinétiques et potentielles
    dxyzs = np.array([get_dxyz(theta,dtheta,phi,dphi) for (theta,dtheta,phi,dphi) in zip(theta_vals,dtheta_vals,phi_vals,dphi_vals)])
    T_vals = np.array([0.5*m*(v[0]**2+v[1]**2+v[2]**2) for v in dxyzs])
    V_vals = np.array([m*g*(l+xyz[2]) for xyz in xyzs])
#Em: énergie mécanique (hamiltonien = T+V), L= Lagrangien (T - V)
    Em_vals = np.array([T + V for (T,V) in zip(T_vals,V_vals)])
    L_vals = np.array([T - V for (T,V) in zip(T_vals,V_vals)])

    for name in statenames:
        States[name] = locals()[name]

    if Debugmode:
        for key in States.keys():
            print(key+": "+str(type(States[key]))+" : "+str(len(States[key])))
        print(f'Acquisition terminée en {round(time()-start_time,6)} secondes', end='\n')
        print("")

#polices utilisées par les fcts d'affichage
font1 = {'family':'serif','color':'blue','size':10}
font2 = {'family':'serif','color':'red','size':10}
font3 = {'family':'serif','color':'green','size':10}
font4 = {'family':'serif','color':'orange','size':10}

#fonction qui affiche, contient des sous-fonctions pour l'animation
def animate_positions():
    """
    Nécessite l'état xyzs dans States
    generate_states(minimal=True)
    
    Anime la trajectoire du pendule en 3D avec matplotlib.animation.Funcanimation
    Selon les valeurs de Showplots et Saveplots, on va afficher/sauvegarder l'animation
    Le nom de fichier est généré automatiquement à partir de la date pour éviter les doublons
    
    Output
    -------
    line : objet muet utilisé par matplotlib, peut se supprimer de la mémoire

    """
    
    start_time = time()
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.set(xlim3d=(-l * 1.1, l * 1.1), xlabel='x')
    ax.set(ylim3d=(-l * 1.1, l * 1.1), ylabel='y')
    ax.set(zlim3d=(-l * 1.1, l * 1.1), zlabel='z')
    line, = ax.plot([], [], [], 'o-',lw = 2)
    plt.suptitle(f'constante de friction: {k} méthode de calcul: {option}',wrap = True)
    try:
        xyzs = States["xyzs"]
        ts = States["times"]
    except KeyError:
        print("Veuillez appeler generate_states(minimal=False) avant d'appeler animate_positions()")
        exit()
        

    if Debugmode:
        print(f'Méthode de calcul: {option}')
        print(f'position initiale: {xyzs[0]}')
        print(f'(theta,phi): {States["theta_vals"][0],States["phi_vals"][0]}')
        print(f'vitesse initiale: {States["dxyzs"][0]}')
        print(f'(dtheta,dphi): {States["dtheta_vals"][0],States["dphi_vals"][0]}')
        print("")

    #Python autorise à définir une fonction locale dans une autre fonction
    def initline():
        line.set_data_3d([0, xyzs[0][0]], [0, xyzs[0][1]], [0, xyzs[0][2]])
        if Debugmode:
            print('initline done')
        return line,

    #pk sa marche pa mdr
    def update(frame,line,xyzs):
        #line contient tous les points à afficher [liste des x],[liste des y]
        #relie automatiquement le centre au pendule
        line.set_data_3d([0,xyzs[frame][0]],[0,xyzs[frame][1]],[0,xyzs[frame][2]])
        #title.set_text('t = %.1fs'%ts[frame])
        #ax.plot(xs[frame],ys[frame],zs[frame])
        #on return tous les objets à animer
        return line,

    ani = FA(fig, update, frames=Nframes,init_func = initline,interval = framerate,fargs = (line,xyzs))

    if Saveplots:
        filename = 'Pendule_simple_3D' + strftime("%d_%m_%Y_%H_%M_%S", localtime()).replace('/', '_') + '.gif'
        ani.save(filename)
        print("fichier "+filename+" sauvegardé")
    if Debugmode:
        print(f'Animation générée en {round(time()-start_time,6)} secondes', end='\n')
        print("")
    if Showplots:
        plt.show()

#Trace énergies cinétique, potentielle, Lagrangien et Hamiltonien
def plot_energies():
    """
    Nécessite les états T_vals,V_vals,Em_vals,L_vals
    generate_states(minimal=True) causera donc une erreur
    
    Génère 2 graphiques pour l'énergie du pendule
    
    1er graphe: Energie cinétique vs Energie potentielle
    2ème graphe: Energie mecanique vs Lagrangien
    Sauvegarde les graphes si Saveplots est à True
    """
    start_time = time()
    fig = plt.figure(figsize=(7,7))
    plt.suptitle(f'constante de friction: {k} méthode de calcul: {option}',wrap = True)
    try:
        Ts = States["T_vals"]
        Vs = States["V_vals"]
        Ems = States["Em_vals"]
        Ls = States["L_vals"]
        ts = States["times"]
    except KeyError:
        print("Veuillez appeler generate_states(minimal=False) avant d'appeler plot_energies()")
        quit()

    #Graphique énergie cinétique et potentielle
    ax1=fig.add_subplot(211)
    ax1.set_title("Energie cinetique", fontdict=font1,loc='left')
    ax1.set_title("Energie potentielle", fontdict=font2,loc='right')
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("E (J)")
    ax1.plot(ts,Ts,color = font1['color'])
    ax1.plot(ts,Vs,color= font2['color'])

    #Graphique énergie mécanique et Lagrangien
    ax2=fig.add_subplot(212)
    ax2.set_title("Energie mécanique", fontdict=font3,loc='left')
    ax2.set_title("Lagrangien", fontdict=font4,loc='right')
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("E (J)")
    ax2.plot(ts,Ems,color=font3['color'])
    ax2.plot(ts,Ls,color=font4['color'])

    if Debugmode:
        print("Graphiques générés en " + str(time() - start_time)[:6] + " secondes", end='\n')
        print("")
    if Saveplots:
        filename = '3DSP_energy' + strftime("%d_%m_%Y_%H_%M_%S", localtime()).replace('/', '_') + '.png'
        plt.savefig(filename)
        print("fichier "+filename+" sauvegardé")
    if Showplots:
        plt.show()


#Modifier pour choper période zenith/periode azimuth
#séparer en 2 get_period et la fct qui plot les angles
def get_periods():
    ts = States["times"]
    thetas = States["theta_vals"]
    dthetas = States["dtheta_vals"]
    phis = States["phi_vals"]
    dphis = States["dphi_vals"]
    zeros_t = [] #zéros de theta
    zeros_p = [] #zéros de phi

    #les zéros sont définis comme les endroits ou x change de signe (valeur à gauche)
    #pour une oscillation amortie on aura la période moyenne des oscillations car la période diminue avec le temps

    #période du zénith (theta)
    for j in range(len(thetas)-1):
        if thetas[j]*thetas[j+1] < 0.0:
            zeros_t.append(ts[j])
    p_t = 2*sum([zeros_t[j+1] - zeros_t[j] for j in range(len(zeros_t) - 1) ])/(len(zeros_t)-1)

    #période de l'azimuth (phi)
    for j in range(len(phis)-1):
        if phis[j]*phis[j+1] < 0.0:
            zeros_p.append(ts[j])
    p_p = 2*sum([zeros_p[j+1] - zeros_p[j] for j in range(len(zeros_p) - 1) ])/(len(zeros_p)-1)

    #affiche tous les [(t,theta(t) et (t+dt, theta(t+dt)) pour toutes les valeurs de t ou theta change de signe
    if Debugmode:
        for z in zeros_t:
            index = np.where(ts == z)[0][0]
            print(f'theta({round(z,3)}) = {thetas[index]}')
            print("")
        for zz in zeros_p:
            index = np.where(ts == z)[0][0]
            print(f'phi({round(zz,3)}) = {phis[index]}')
            print("")
        print(f'{len(zeros_t)} changements de signe de theta trouvés')
        print(f'période de theta: {p_t}')
        print(f'{len(zeros_p)} changements de signe de phi trouvés')
        print(f'période de phi: {p_p}')

    #A refactorer dans une fonction à part, on a aussi phi et dphi à plotter.
    if Showplots:
        fig = plt.figure(figsize = (7,7))
        plt.suptitle(f'constante de friction: {k} méthode de calcul: {option}', wrap=True)

        ax1 = plt.subplot(211)
        ax1.set_title("theta (rad)", fontdict=font1, loc='left')
        ax1.set_xlabel("t (s)")
        ax1.plot(ts, thetas, color=font1['color'])

        ax2 = plt.subplot(212)
        ax2.set_title("dtheta (rad/s)", fontdict=font2, loc='right')
        ax2.set_xlabel("t (s)")
        ax2.plot(ts, dthetas, color=font2['color'])

        plt.show()

    return (p_t,p_p)
