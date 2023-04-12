import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
from time import time, localtime, strftime
from inspect import isfunction

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
dtheta_0 = 0.0

#TRES TRES ATTENTION: Python transforme automatiquement ces string en fonctions
#La partie du code qui permet de choisir sa fct de calcul marche bien 
#Par contre l'affichage met <function machin at truc> sur le graphe...
option: str = "Euler"
options = ["Euler","RK2","RK4"]

        
#Constantes qu'on peut modifier depuis le main()
constants = ["Debugmode", "Saveplots","Showplots", "Nframes", "tmax", "framerate", "theta_0", "dtheta_0", "l", "m","k","option"]
def set_constants(**kwargs):
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
        
    Output
    -------
    None.
    """
    start_time = time()
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

#fonction qui transforme S en dS/dt
#si on veut des valeurs + précises on peut faire un schéma de runge-kutta
def diff_S(S,t,g,l,k):
    """    
    Fonction qu'on rentre dans le schéma d'intégration
    
    Arguments
    ----------
    S = [theta,dtheta] : etat du système à l'instant t
    t : float: temps
    g, l, k: float: constantes

    Output
    -------
    dS = [dtheta,ddtheta]: dérivée de l'etat à l'instant t
    ddtheta provient de l'équadiff du système

    """
    theta,dtheta = S
    return np.array([dtheta,-g*np.sin(theta)/l - k*dtheta/m])

#Attention: RK2 et RK4 bug (énergie mécanique augmente quand il n'y a pas de frottement)
#on prend le même formalisme que scipy.odeint
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
        y[i + 1] = y[i] + h * func(y[i] + func(y[i], t[i], *args) * h / 2., t[i] + h / 2., *args)
    return y

def RungeKutta4(func, y0, t, args):
    """
    résout l'équation dy/dt = f(t,y(t),*args)
    
    Parameters
    ----------
    func : fonction qui transforme S en dS/dt
    y0 : état du système à t = 0 = (theta0,phi0)
    t : array contentant tous les instants 
    args : constantes de l'equation: (g,l,k) dans notre cas
    Returns
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
#liste de toutes les variables d'état
statenames = ["theta_vals","dtheta_vals","x_vals","y_vals","dx_vals","dy_vals","T_vals","V_vals","Em_vals","L_vals","times"]
#dictionnaire des valeurs de chaque état pour tout t, utilisé par tout le reste du programme
States = {}
#fonction principale du programme qui remplit le dictionnaire States
#calcule en rentrant diff_S dans scipy.odeint qui fait un schéma d'euler par défaut visiblement
#option 'minimal' pour ne générer que theta et dtheta (utile pour faire des simu en boucle)
def generate_states(minimal: bool):
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
    
    2/Etats calculés seulement si minimal est à False
    -x_vals,y_vals: valeurs des coordonnées cartésiennes
    -dx_vals, dy_vals: valeurs des dérivées de x et y
    -T_vals: valeurs de l'énergie cinétique (0.5*m*v^2)
    -V_vals: valeurs de l'énergie potentielle (m*g*(l+z))
    -L_vals: valeurs du lagrangien T - V
    -Em_vals: valeurs d'énergie mécanique T + V (hamiltonien)
    
    Arguments
    ----------
    minimal : bool
    Si mis à True, on ne calcule que les états indispensables (utile pour enchainer plusieurs simulations)
    Certaines fct d'affichage nécessitent minimal à False (voir doc)'

    Output
    -------
    Ne retourne rien mais met à jour le dictionnaire States

    """
    method = odeint
    if option =='RK2':
        method = RungeKutta2
    elif option == 'RK4':
        method = RungeKutta4
    
    start_time = time()
    if Debugmode:
        print("")
        print("Ce code simule un pendule 2D avec scipy")
        print(f'Méthode de calcul: {method.__name__}')
        print(f'masse du pendule: {m}')
        print(f'longueur du fil: {l}')
        print(f'constante de friction: {k}')
        print(f'etat initial: theta = {round(theta_0,6)}  dtheta = {round(dtheta_0,6)}')
        print("")

    S_zero = [theta_0, dtheta_0]
    global States

#Contient la liste des (theta,dtheta) pour tous t
    times = np.linspace(0, tmax, Nframes)
    phasestates = np.zeros((len(times),len(S_zero)))
    try:
        phasestates =method(diff_S,S_zero,times,args = (g,l,k))
    except Exception:
        raise NameError(f'{method.__name__} ne génère pas correctement les états')
        print(f'méthodes de calcul disponibles: {options}')

#on extrait la liste des valeurs de theta et dtheta
    theta_vals = phasestates[:,0]
    dtheta_vals = phasestates[:,1]
    if minimal:
        States = {"theta_vals": theta_vals, "dtheta_vals": dtheta_vals,"times": times}
        return None
#on calcule les positions cartésiennes pour l'animation
    x_vals = l*np.sin(theta_vals)
    y_vals = -l*np.cos(theta_vals)
#On calcule les énergies cinétiques et potentielles
#Peut se refactorer plus joliment avec zip()
    dx_vals = np.array([l*dtheta_vals[j]*np.cos(theta_vals[j]) for j in range(Nframes)])
    dy_vals = np.array([l*dtheta_vals[j]*np.sin(theta_vals[j]) for j in range(Nframes)])
    T_vals = np.array([0.5*m*(dx_vals[j]**2 + dy_vals[j]**2) for j in range(Nframes)])
    V_vals = np.array([m*g*(l+val) for val in y_vals])
#Em: énergie mécanique (hamiltonien = T+V), L= Lagrangien (T - V)
    Em_vals = np.array([T_vals[j]+V_vals[j] for j in range(Nframes)])
    L_vals = np.array([T_vals[j]-V_vals[j] for j in range(Nframes)])

    for name in statenames:
        try:
            States[name] = locals()[name]
        except Exception:
            raise NameError(name+" n'est pas une variable d'etat")
    if Debugmode:
        for key in States.keys():
            print(key+": "+str(type(States[key]))+" : "+str(len(States[key])))
        print(f'generate_states fini en {round(time()-start_time,6)} s', end='\n')
        print("")


#polices utilisées par les fcts d'affichage
font1 = {'family':'serif','color':'blue','size':10}
font2 = {'family':'serif','color':'red','size':10}
font3 = {'family':'serif','color':'green','size':10}
font4 = {'family':'serif','color':'orange','size':10}
nbfonts = 4
fonts = [globals()['font'+str(i)] for i in range(1,nbfonts+1)]

#anime la trajectoire du pendule
def animate_positions():
    """
    Nécessite les états x_vals et y_vals
    generate_states(minimal=True) causera donc une erreur
    
    Anime la trajectoire du pendule en 3D avec matplotlib.animation.Funcanimation
    Selon les valeurs de Showplots et Saveplots, on va afficher/sauvegarder l'animation
    Le nom de fichier est généré automatiquement à partir de la date pour éviter les doublons
    
    
    Output
    -------
    line : objet muet utilisé par matplotlib, peut se supprimer de la mémoire

    """
    start_time = time()
    fig=plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], 'o-')
    ax.set_aspect('equal')
    title = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    plt.suptitle(f'friction: k = {k} , méthode de calcul: {option}',wrap = True)
    plt.xlabel("x")
    plt.ylabel("y")
    try:
        xs = States["x_vals"]
        ys = States["y_vals"]
        ts = States["times"]
    except KeyError:
        print("Veuillez appeler generate_states(minimal=False) avant d'appeler animate_positions()")
        exit()

    #Python autorise à définir une fonction locale dans une autre fonction
    #On pourrait les définir en global mais elles demanderaient trop de paramètres pour rentrer dans FuncAnimation
    def initline():
        ax.set_xlim(-l*1.2,l*1.2)
        ax.set_ylim(-l*1.2,l*1.2)
        return line,

    def update(frame):
        #line contient tous les points à afficher [liste des x],[liste des y]
        #relie automatiquement le centre au pendule
        line.set_data([0,xs[frame]], [0,ys[frame]])
        title.set_text('t = %.1fs'%ts[frame])
        #on return tous les objets à animer
        return line,title

    #cf doc de matplotlib.animation.FuncAnimation
    ani = FA(fig, update, frames=len(ts),init_func = initline,blit=True,interval = framerate)

    if Saveplots:
        filename = 'Pendule_simple_2D' + strftime("%d_%m_%Y_%H_%M_%S", localtime()).replace('/', '_') + '.gif'
        ani.save(filename)
        print("fichier "+filename+" sauvegardé")

    if Debugmode:
        print(f"animate_positions fini en {round(time()-start_time,6)} s", end='\n')
        print("")

    if Showplots:
        plt.show()

def plot_energies():
    """
    Nécessite les etats T_vals,V_vals,Em_vals,L_vals
    generate_states(minimal=True) causera donc une erreur
    
    Génère 2 graphiques pour l'énergie du pendule
    
    1er graphe: Energie cinétique vs Energie potentielle
    2ème graphe: Energie mecanique vs Lagrangien
    """
    start_time = time()
    fig2 = plt.figure(figsize=(7,7))
    plt.suptitle(f'friction: k = {k} , méthode de calcul: {option}',wrap = True)
    try:
        Ts = States["T_vals"]
        Vs = States["V_vals"]
        Ems = States["Em_vals"]
        Ls = States["L_vals"]
        ts = States["times"]
    except KeyError:
        print("Veuillez appeler generate_states(minimal=False) avant d'appeler animate_positions()")
        exit()
    
    #Graphique énergie cinétique et potentielle
    ax1=fig2.add_subplot(211)
    ax1.set_title("Energie cinetique", fontdict=font1,loc='left')
    ax1.set_title("Energie potentielle", fontdict=font2,loc='right')
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("E (J)")
    ax1.plot(ts,Ts,color = font1['color'])
    ax1.plot(ts,Vs,color= font2['color'])

    #Graphique énergie mécanique et Lagrangien
    ax2=fig2.add_subplot(212)
    ax2.set_title("Energie mécanique", fontdict=font3,loc='left')
    ax2.set_title("Lagrangien", fontdict=font4,loc='right')
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("E (J)")
    ax2.plot(ts,Ems,color=font3['color'])
    ax2.plot(ts,Ls,color=font4['color'])

    if Debugmode:
        print(f"plot_energies fini en {round(time()-start_time,6)} s", end='\n')
        print("")
    if Saveplots:
        filename = '2DSP_energy_'+strftime("%d_%m_%Y_%H_%M_%S", localtime())+'.png'
        plt.savefig(filename)
        print(f"fichier {filename} sauvegardé")
    if Showplots:
        plt.show()

#retourne une estimation rudimentaire de la période du pendule
#trace le graphe de theta et dtheta si Showplots est à True
def get_period():
    """
    Nécessite uniquement les états times et thetas
    On peut donc appeler generate_states(minimal=True) sans problème
    
    Estime la période expérimentale du pendule avec la méthode suivante:
    -On note les valeurs de t où theta change de signe
    -On note la moyenne des écarts entre deux valeurs successives
    -On divise par 2 car il y a 2 changements de signe par période
    -On return la valeur notée p (float)
    
    Si Showplots est à True, on calcule:
    -Pulsation libre: omegal = sqrt(g/l) 
    -Pulsation amortie: omega = sqrt(omega² - (k/2m)²)
    -Période libre: pl = 2pi/omegal
    -Période amortie: pth = 2pi/omega
    
    On trace ensuite un graphe avec:
    -theta(t): angle expérimental obtenu par simulation
    -cos(omega*t): signal suivant la période théorique
    
    Dans ce cas, on return également la période théorique
    
    Output
    -------
    p : float: période expérimentale du pendule si Showplots = False
    (p,pth) si Showplots = True
    """
    start_time = time()
    ts = States["times"]
    thetas = States["theta_vals"]
    zeros = []
    #on précalcule quelques valeurs pour les fct d'affichage
    
    #les zéros sont définis comme les endroits ou theta change de signe (valeur à gauche)
    #pour une oscillation amortie on aura la période moyenne des oscillations
    for j in range(len(thetas)-1):
        if thetas[j]*thetas[j+1] < 0.0:
            zeros.append(ts[j])
            
    p = 2*sum([zeros[j+1] - zeros[j] for j in range(len(zeros) - 1) ])/(len(zeros)-1)
    pth=0.0 #n'est pas utilisée si Showplots = False
    
    
    if Debugmode:
        for z in zeros:
            index = np.where(ts == z)[0][0]
            print("theta("+str(round(z,3))+") = "+str(thetas[index]))
            #print("theta(" + str(round(ts[index + 1], 3)) + ") = " + str(thetas[index + 1]))
            print("")
        print(str(len(zeros))+" changements de signe de theta trouvés")
        
    if Showplots:
        omegal = np.sqrt(g/l) #pulsation libre
        omega = omegal if k <= 0.0001 else np.sqrt(abs(omegal**2 - (k/(2*m))**2)) #pulsation amortie
        pl = 2*np.pi/omegal #période libre
        pth = 2*np.pi/omega #période amortie
        reference = 0.5*max(thetas)*np.cos(omega*ts)
        if Debugmode:
            print(f'période expérimentale: {p}')
            print(f'période propre = {pl}')
            print(f'pulsation propre = {omegal}')
            print(f'période amortie = {pth}')
            print(f'pulsation amortie = {omega}')
           
        fig = plt.figure(figsize = (7,7))
        plt.suptitle(f'friction: k = {k} période: {round(pth,6)} , méthode de calcul: {option}', wrap=True)
        ax = fig.add_subplot(111)
        ax.set_title("theta(t) (rad)", fontdict=font1, loc='left')
        ax.set_title(f'cos({round(omega,3)}t)', fontdict=font2, loc='right')
        ax.set_xlabel("t (s)")
        ax.plot(ts, thetas, color=font1['color'])
        ax.plot(ts, reference, color=font2['color'])
    if Saveplots:
        filename = '2DSP_period_'+strftime("%d_%m_%Y_%H_%M_%S", localtime())+'.png'
        plt.savefig(filename)
        if Debugmode:
            print(f"get_period done in {round(time() - start_time, 6)} s", end='\n')
            print("")
        plt.show()

    #un peu dangereux, bien faire attention à l'utilisation
    return (p,pth) if Showplots else p

def animate_phasespace(**kwargs):
    """
    Anime la trajectoire du pendue dans l'espace des phases (theta,dtheta):
    Utilise les états times, theta_vals et dtheta_vals
    Peut donc s'appeler avec generate_states(minimal=True)
        
    Arguments
    ----------
    **kwargs : {trail: bool} (dictionnaire d'options')
    détermine si on laisse toute la trajectoire à l'écran où non
    plusieurs options de plot seront ajoutées à l'avenir

    """
    start_time = time()
    ts = States["times"]
    thetas = States["theta_vals"]
    dthetas = States["dtheta_vals"]
    
    fig = plt.figure(figsize = (6,6))
    plt.suptitle(f'friction: k = {k} , méthode de calcul: {option}',wrap = True)
    ax = fig.add_subplot(111)
    ax.set_title("Trajectoire dans l'espace des phases", fontdict=font1)
    title = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    ax.set_xlabel("theta")
    ax.set_ylabel("dtheta")
    
    line, = ax.plot([], [], '')

    def initline():
        ax.set_xlim(min(thetas)*1.1,max(thetas)*1.1)
        ax.set_ylim(min(dthetas)*1.1,max(dthetas)*1.1)
        return line,
    
    def update_trail(frame):
        #line contient tous les points à afficher [liste des x],[liste des y]
        #relie automatiquement le centre au pendule
        line.set_data([thetas[:frame]], [dthetas[:frame]])
        title.set_text('t = %.1fs'%ts[frame])
        #on return tous les objets à animer
        return line,title
    
    def update_notrail(frame):
        #line contient tous les points à afficher [liste des x],[liste des y]
        #relie automatiquement le centre au pendule
        line.set_data([thetas[int(0.9*frame):frame]], [dthetas[int(0.9*frame):frame]])
        title.set_text('t = %.1fs'%ts[frame])
        #on return tous les objets à animer
        return line,title
    
    if 'trail' in kwargs.keys() and kwargs['trail'] == False:
        updatefunc = update_notrail
    else:
        updatefunc = update_trail
        
    ani = FA(fig, updatefunc, frames=len(ts),init_func = initline,blit=True,interval = framerate)
    if Saveplots:
        filename = 'Phase_Pendule2D_' + strftime("%d_%m_%Y_%H_%M_%S", localtime()).replace('/', '_') + '.gif'
        ani.save(filename)
        print("fichier "+filename+" sauvegardé")
    if Debugmode:
        print(f"plot_phasespace done in {round(time() - start_time, 6)} s", end='\n')
        print("")
    plt.show()


#graphe la période théorique et expérimentale pour différentes valeurs de theta_0
def small_angle_errors(N,thetazero,thetamax):
    """
    Effectue N simulations ou theta(0) varie entre thetazero et thetamax
    Pour chaque simulation, note l'écart entre période théorique 
    donnée par l'approx. des petits angles et période expérimentale donnée par get_period()'
    Trace le graphe des erreurs absolue et relative
    Sauvegarde le graphe si Saveplots est à True
    
    Appelle les fonctions set_constants, generate_states et get_period

    Parameters
    ----------
    N : nb de points sur le graphe
    thetazero : theta(0) sur la 1ere simulation
    thetamax : theta(0) sur la n-ième simulation

    Returns
    -------
    None.

    """
    print_stuff = Debugmode #on sauve le debugmode avant de le mettre à false
    #on remet le Debugmode à false sinon toutes les fct de calcul vont afficher des trucs
    set_constants(Showplots = False, Debugmode = False, Saveplots = False)
    start_time = time()
    step = (thetamax - thetazero)/N
    angles = []
    abs_errs = []
    rel_errs = []
    Tl = 2*np.pi*np.sqrt(l/g) #période d'oscillations libres (sans frottement)
    alpha = k/(2*m) #coeff de frottement
    Tc = 2*np.pi/np.sqrt(abs(g/l - alpha**2))
    
    #l'indice n'a pas le droit de s'appeler k où l (constantes du programme)
    for i in range(N):
        angles.append(thetazero+step*i)
        set_constants(theta_0 = thetazero + step*i)
        generate_states(minimal = True)
        p = get_period()
        abs_errs.append(abs(Tl-p))
        rel_errs.append(abs(Tl-p)/p)

        if print_stuff:
            print(f"angle initial = {angles[-1]} rad")
            print(f"période propre: {Tl} rad/s")
            print(f"période amortie: {Tc} rad/s")
            print(f"période expérimentale: {p} rad/s")
            print(f"erreur absolue = {abs_errs[-1]}")
            print(f"erreur relative = {rel_errs[-1]}")
            print("")

    fig,ax1 = plt.subplots()
    ax1.plot(angles,abs_errs,color='red')
    ax1.set_xlabel("angle initial (rad)")
    ax1.set_ylabel("erreur absolue (Hz)",color='red')

    ax2 = ax1.twinx()
    ax2.plot(angles,rel_errs,color='blue')
    ax2.set_ylabel("erreur relative",color='blue')

    fig.suptitle("erreur d'approximation des petits angles")
    fig.tight_layout()

    if Saveplots:
        filename = 'approx_err' + strftime("%d_%m_%Y_%H_%M_%S", localtime()) + '.png'
        plt.savefig(filename)
        print(f'fichier {filename} sauvegardé')
    if print_stuff:
        print(f'small_angle_error done in {round(time() - start_time, 6)} s', end='\n')
        print('')

    plt.show()

#donne le nom de toutes les constantes du programme
def print_variables():
    print("Nom des constantes modifiables via set_constants(): ",end='\n')
    print(str(constants),end='\n')
    print("")
    print("Nom des variables d'état du système stockées dans le dict States: ")
    print(str(statenames),end='\n')
    print("")