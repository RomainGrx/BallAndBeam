# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 22-02-2020

import numpy as np
from Simulator import Simulator


class BBSimulator(Simulator):
    """
    Classe qui represente un simulateur "general" pour un systeme ball and beam decrit comme suit:
        - Vecteur d'etat      : [position, vitesse]
        - Vecteur de commande : [angle]  (cet angle est encore assez abstrait pour cette classe)
        - Vecteur de sortie   : [position]

    On assigne aussi des valeurs a plusieurs parametres caracteristiques du systeme Ball and Beam du P4 MAP.

    Note: La commande etant l'angle, on peut le faire varier de maniere instantanee avec ce simulateur.
          Cependant, dans un systeme reel, il y a un leger delai du a l'inertie de la poutre, notamment.
          Il faut garder cela en tete lors des tests et de la conception de la loi de commande.

    Note: Certains parametres du modele ont ete obtenus par optimisation experimentale et pourraient sembler
          incoherents. Par exemple, le moment d'inertie de la bille ne semble pas dependre directement de la
          masse de celle-ci, ou bien la masse de la bille est plus du double des 55.9g que fait la bille.
          Cela s'explique par le fait que le modele ne tient pas compte de plusieurs phenomenes tels que
          les turbulences de l'eau, l'inertie de la poutre, etc, et que tous ces phenomenes non pris en compte
          se retrouvent pris en compte de maniere implicite dans la masse de la bille, son moment d'inertie, etc.
          Par exemple, un parametre 'm' superieur a 55.9g traduit le fait qu'il y a une masse d'eau qui affecte
          le deplacement de la bille, etc.
    """
    def __init__(self, dt=0.05, buffer_size=10000):
        params = {
            "m": 4.07577970e-01,     # Masse de la bille [kg]
            "r": 4.50930559e-02,     # Rayon de la bille [m]
            "g": 8.90574462e+01,     # Acceleration gravitationnelle a basse altitude [m/s^2]
            "rho": 2.87279734e+00,   # Masse volumique de l'eau [kg/m^3]
            "l": 0.775,              # Longueur de la poutre [m]
            "d": 55 / 1000,          # Longueur de la premiere barre attachee au servo [m]
            "b": 150 / 1000,         # Distance entre le pivot et le point d'attache du second bras du servo [m]
            "kf": 9.12765693e+01,    # Coefficient de frottement obtenu par optimisation experimentale []
            "jb": 5.08989712e-02,    # Moment d'inertie de la bille obtenu par optimisation experimentale [kg*m^2]
        }

        params["v"] = 4/3 * np.pi * params["r"] ** 3  # Volume de la bille [m^3]
        n_states, n_commands, n_outputs = 2, 1, 1
        super().__init__(params, n_states, n_commands, n_outputs, dt, buffer_size)

    def y(self):
        return self.all_x[self.timestep, 0]


class BBSimpleSimulator(BBSimulator):
    """
    Simulateur tres simplifie d'un Ball and Beam. Il simule un ball and beam classique (sans eau), sans frottements
    et sans tenir compte des limites de la poutre (la poutre a une longueur "infinie"). La commande correspond a
    l'angle de la poutre (et donc pas du servo).
    """
    def __init__(self, dt=0.05, buffer_size=10000):
        super().__init__(dt, buffer_size)

    def dxdt(self):
        m, x, g, r = self.params["m"], self.all_x[self.timestep], self.params["g"], self.params["r"]
        jb, alpha, dalpha_dt = self.params["jb"], self.all_u[self.timestep], self.dudt()
        dx1_dt = x[1]
        dx2_dt = ((m * x[0] * dalpha_dt ** 2 - m * g * np.sin(alpha)) / (jb / r ** 2 + m))[0]
        return np.array([dx1_dt, dx2_dt])


class BBAlphaSimulator(BBSimulator):
    """
    Simulateur d'un ball and beam, dans lequel la commande correspond a l'angle de la poutre (alpha) et donc
    pas a l'angle du servo (theta). On tient compte des frottemtents fluides en les approximant par une fonction
    lineaire de la vitesse. On tient compte de la poussee d'Archimede. On tient compte des limites de la poutre
    (quand la balle arrive en bout de tube, elle bute). Le bridage de la commande ne depend pas de la classe mais
    de la fonction 'command_func' qui est passee en argument a 'simulate'.
    """
    def __init__(self, dt=0.05, buffer_size=10000):
        super().__init__(dt, buffer_size)

    def dxdt(self):
        m, x, g, r = self.params["m"], self.all_x[self.timestep], self.params["g"], self.params["r"]
        jb, alpha, dalpha_dt = self.params["jb"], self.all_u[self.timestep], self.dudt()
        rho, v, kf = self.params["rho"], self.params["v"], self.params["kf"]
        dx1_dt = x[1]
        dx2_dt = ((m * x[0] * dalpha_dt ** 2 - np.sin(alpha) * (m - rho * v) * g - kf * x[1]) / (jb / r ** 2 + m))[0]
        return np.array([dx1_dt, dx2_dt])

    def update_state(self):
        # Ajout d'un mecanisme de "bridage" de la position: la balle est contrainte a rester dans le tube.
        new_x = self.all_x[self.timestep] + self.dt * self.dxdt()
        if new_x[0] > self.params["l"] / 2:
            # Position imposee et vitesse mise a zero
            self.all_x[self.timestep + 1, 0] = self.params["l"] / 2
            self.all_x[self.timestep + 1, 1] = 0
        elif new_x[0] < -self.params["l"] / 2:
            # Position imposee et vitesse mise a zero
            self.all_x[self.timestep + 1, 0] = -self.params["l"] / 2
            self.all_x[self.timestep + 1, 1] = 0
        else:
            self.all_x[self.timestep + 1] = new_x

    def simulate(self, command_func, command_noise_func=lambda *args, **kwargs: 0,
                 output_noise_func=lambda *args, **kwargs: 0, n_steps=np.inf, init_state=np.zeros((2,))):
        # Ajout d'un mecanisme de verification de la validite de la position initiale.
        if np.size(init_state) == 1 and np.abs(init_state) > self.params["l"] / 2 or\
                np.size(init_state) == self.n_states and np.abs(init_state[0]) > self.params["l"] / 2:
            raise ValueError("Initial position is not on the beam")
        return super().simulate(command_func, command_noise_func, output_noise_func, n_steps, init_state)


class BBThetaSimulator(BBAlphaSimulator):
    """
    Sous-classe de BBAlphaSimulator. Changements par rapport a cette classe-la:
        1) La commande se fait sur l'angle theta (servo) au lieu de l'angle alpha (poutre)

        2) Il y a une gestion du decalage entre angle commande et angle obtenu que l'on peut observer en realite.
           Ce decalage est modelise par un parametre constant 'theta_offset', tel que, quand on commande un angle
           'theta', l'angle reel du servo est de 'theta' + 'theta_offset'. 'theta_offset' a une valeur d'environ
           -6.17 deg par defaut, car c'est ce que l'on a pu observer en laboratoire et ce qui correspond le mieux
           aux donnees experimentales.

        3) Il y a une gestion du frottement statique. Elle est modelisee avec deux parametres 'stat_bound' et
           'stat_spd_coeff', qui sont tels que la balle sera forcee a s'arreter quand la condition suivante est
           verifiee:
               abs('theta') + 'stat_spd_coeff' / 'stat_bound' * abs('speed') < 'stat_bound'
           Ces parametres s'interpretent ainsi:
               - Si la balle est a l'arret et que abs('theta') < 'stat_bound', alors la balle reste a l'arret;
               - Si la valeur de 'stat_spd_coeff' est egale a 'v', alors, a angle nul, la bille s'arretera quand la
                 valeur absolue de sa vitesse est inferieure a 'v'.
           Les valeurs par defaut de ces parametres sont obtenues de maniere experimentale

        4) Il y a une gestion de la non-linearite des forces de frottement par rapport a la norme de la vitesse.
           On modelise cela avec un parametre 'ff_pow' dont la valeur est obtenur suite a une optimisation sur des
           donnees experimentales.

        Note: certains de ces parametres en cachent d'autres. Il n'est pas anormal d'observer des forces de
              de frottement super-lineaires quand on sait que les phenomenes d'ecoulement turbulent (etc.) n'ont
              pas ete rigoureusement modelises.

        Note: L'offset est pris en compte dans 'all_u'.
    """

    def __init__(self, dt=0.05, buffer_size=10000):
        super().__init__(dt, buffer_size)
        # Parametre pour la gestion de l'offset de l'angle
        self.params["theta_offset"] = -1.22168637e-01  # Quand on commande theta = 0 deg, on aura theta = -7 deg

        # Parametres pour la simulation du frottement statique
        self.params["stat_bound"] = 0.12272812  # Vitesse nulle: la bille s'arrete quand abs(theta) < ~7deg
        self.params["stat_spd_coeff"] = 0.036  # A angle 0, la balle s'arrete si v < 0.036 m/s

        # Parametre pour la gestion de la non-linearite du frottement par rapport a la vitesse
        self.params["ff_pow"] = 1.57403518e+00  # Les frottements dependent de la vitesse ** ff_pow

    def dxdt(self):
        stat_bound, stat_spd_coeff = self.params["stat_bound"], self.params["stat_spd_coeff"]
        m, x, g, r = self.params["m"], self.all_x[self.timestep], self.params["g"], self.params["r"]
        d, b, l, jb = self.params["d"], self.params["b"], self.params["l"], self.params["jb"]
        theta, dtheta_dt = self.all_u[self.timestep], self.dudt()
        alpha = np.arcsin(d / b * np.sin(theta))
        dalpha_dt = d * np.cos(theta) * self.dudt() / (l * np.sqrt(1 - (d * np.sin(theta) / l) ** 2))
        rho, v, kf, ff_pow = self.params["rho"], self.params["v"], self.params["kf"], self.params["ff_pow"]
        x1_pow = np.power(np.abs(x[1]), ff_pow) * np.sign(x[1])  # Laisser le abs sinon racines complexes
        dx1_dt = x[1]
        dx2_dt = ((m * x[0] * dalpha_dt ** 2 - np.sin(alpha) * (m - rho * v) * g - kf * x1_pow) / (jb / r ** 2 + m))[0]

        # Gestion du frottement statique
        if abs(theta) + stat_bound / stat_spd_coeff * abs(dx1_dt) < stat_bound:
            dx1_dt = 0
            dx2_dt = 0

        return np.array([dx1_dt, dx2_dt])

    def simulate(self, command_func, command_noise_func=lambda *args, **kwargs: 0,
                 output_noise_func=lambda *args, **kwargs: 0, n_steps=np.inf, init_state=np.zeros((2,))):
        # Prise en compte de l'offset dans la commande. En faisant ainsi, l'offset ne sera pas pris en compte
        # dans le bridage de la commande (e.g. une commande bridee entre 'low' et 'up' deviendra bridee
        # entre 'low' + 'offset' et 'up' + 'offset')
        return super().simulate(lambda *args, **kwargs: command_func(*args, **kwargs) + self.params["theta_offset"],
                                command_noise_func, output_noise_func, n_steps, init_state)


class BBObj7Simulator(BBThetaSimulator):
    """
    Simulateur pour le Ball and Beam qui se base sur 'BBThetaSimulator'. L'ajout qui est fait dans cette class-ci est de
    permettre d'utiliser une fonction 'perturbation' afin d'agir sur l'acceleration. Cette fonction est decrite comme
    suit:
            a = perturbation(a_desired, x, v, alpha, t)

            Fonction de perturbation qui permet de forcer l'acceleration de la bille. L'acceleration non-forcee est
            'a_desired'. Pour rendre cette fonction sans effet, faire 'return a_desired'.

            :param a_desired : Valeur de l'acceleration de la bille issue du modele physique [m/s^2]
            :param x         : Position de la bille [m]
            :param v         : Vitesse de la bille [m/s]
            :param alpha     : Angle du servo (pas de la poutre!) [rad]
            :param t         : Temps depuis le debut de la simulation [s]
            :return          : Valeur de l'acceleration forcee de la bille [m/s^2]

            Note: Ici, on travaille en unites SI. Ce n'est pas le cas dans la fonction d'interface du fichier
                  'Interface.py', dans lequel on travaille en cm et en deg, comme demande dans les consignes.

    Cette fonction de perturbation est l'unique modification faite au simulateur 'BBThetaSimulator'. Elle doit etre
    utilisee avec precaution car elle a une priorite maximale. Il est donc possible de modifier completement la
    dynamique du systeme avec cette fonction, rendant le simulateur 'BBThetaSimulator' inutile.
    """
    def __init__(self, perturbation=lambda *args: args[0], dt=0.05, buffer_size=10000):
        super().__init__(dt, buffer_size)
        self.perturbation = perturbation

    def dxdt(self):
        # Application de la fonction de perturbation pour le calcul de l'acceleration de la bille
        dx1_dt, dx2_dt = super().dxdt()
        pos, spd = self.all_x[self.timestep]
        theta = self.all_u[self.timestep]
        t = self.timestep * self.dt
        return np.array([dx1_dt, self.perturbation(dx2_dt, pos, spd, theta, t)])


# La suite du code ne sera executee que si le fichier 'BBSimulators.py' est lance directement. Elle ne le sera pas si ce
# fichier est utilise comme import dans un autre fichier. La section ci-dessous sert de code de demonstration pour
# l'utilisation des classes implementees ci-dessus.
#
# Note importante: ci-dessous, on retrouve des tests des differents simulateurs. Un simulateur n'est pas un controleur
# et il ne faut donc pas s'attendre a voir des artifices de controle tels que de l'idiot-proofing ici. Le controle se
# passe plus tard, dans le fichier 'BBControllers.py'.
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as sig       # Pratique pour generer des signaux carres

    # Fonction de commande dont la sortie est bridee entre -50deg et +50deg.
    # Rappel: le retour de cette fonction est exprime en radians -> utiliser np.deg2rad
    @Simulator.command_limiter(low_bound=np.deg2rad(-50), up_bound=np.deg2rad(50))
    def my_command_func(timestep, params, all_t, all_u, all_y, dt):
        # Donnees utilisables jusqu'a l'indice [timestep - 1]
        t = timestep * dt

        # Decommenter les diverses lois de commande pour observer des comportements differents.

        # Commande nulle constante
        # return 0

        # Commande constante a +50deg
        # return np.deg2rad(50)

        # Commande de sinus d'amplitude a [rad] et de periode p [s]
        # a, p = np.deg2rad(50), 10
        # return np.sin(t * 2 * np.pi / p) * a

        # Commande de signal carre d'amplitude a [rad], de periode p [s] et de duty_cycle d [%]
        a, p, d = np.deg2rad(50), 10, 50
        return sig.square(t * 2 * np.pi / p, duty=d/100) * a

    # Fonction de bruit sur la commande
    def my_command_noise_func(timestep, params, all_t, all_u, all_y, dt):
        # Decommenter pour tester diverses fonctions de bruit

        # Pas de bruit: commande parfaitement fidele
        return 0

        # Erreur uniformement distribuee entre -3deg et +3deg lors de l'application de la commande
        # return (2 * np.random.random(1) - 1) * np.deg2rad(3)

    # Fonction de bruit sur la mesure
    def my_output_noise_func(timestep, params, all_t, all_u, all_y, dt):
        # Decommenter pour tester diverses fonctions de bruit

        # Pas de bruit: commande parfaitement fidele
        return 0

        # Erreur uniformement distribuee entre -2.5cm et +2.5cm lors de la mesure
        # return (2 * np.random.random(1) - 1) * 0.025  # Erreur de +- 2.5cm a la mesure

    # Fonction de perturbation, pour l'objectif 7
    # Rappel: ici on travaille avec les unites SI (m et rad)
    # Attention: le parametre s'appelle 'alpha' uniquement parce que c'etait demande dans les consignes. Il s'agit bien
    # de l'angle du servo et non pas de la poutre. Pour etre coherent avec nos notations, il aurait fallu l'appeler
    # theta.
    def my_perturbation(a_desired, x, v, alpha, t):
        # Decommenter pour tester diverses perturbations

        # Pas de perturbation
        return a_desired

        # Perturbation d'une duree d'une seconde, au debut de la simulation
        # if 0 <= t < 1:
        #     return a_desired * 1.10
        # else:
        #     return a_desired

    # Decommenter pour tester les differents simulateurs (complexite croissante)
    # Celui utilise dans le projet est le 'BBThetaSimulator', sauf pour l'objectif 7 ou il s'agit du 'BBObj7Simulator'

    # sim = BBSimpleSimulator(dt=0.05, buffer_size=1000)
    # sim = BBAlphaSimulator(dt=0.05, buffer_size=1000)
    # sim = BBThetaSimulator(dt=0.05, buffer_size=1000)
    sim = BBObj7Simulator(my_perturbation, dt=0.05, buffer_size=1000)

    # Decommenter pour choisir un etat initial [position, vitesse] (unites SI)
    my_init_state = np.array([0, 0])
    # my_init_state = np.array([0, -0.333])
    # my_init_state = np.array([1, 0])

    # Lancement de la simulation
    sim.simulate(my_command_func, my_command_noise_func, my_output_noise_func, init_state=my_init_state)

    # Creation d'un graphe pour afficher le resultat de la simulation
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True)

    ax1.plot(sim.all_t, sim.all_y, "k-", linewidth=2, label="Simulated position")
    ax1.plot(sim.all_t, np.full(sim.all_t.shape, -sim.params["l"] / 2), "r--", label="Position bounds")
    ax1.plot(sim.all_t, np.full(sim.all_t.shape, sim.params["l"] / 2), "r--")

    # Afficher la commande ideale voulue, celle qui correspond a l'angle qu'aura le servo
    ax2.plot(sim.all_t, np.rad2deg(sim.all_u - sim.params["theta_offset"]), "k-", linewidth=2, label="Command")

    # Afficher la commande avec offset, permettant de compenser les imperfections du servo
    # ax2.plot(sim.all_t, np.rad2deg(sim.all_u), "k-", linewidth=2, label="Command")

    ax2.plot(sim.all_t, np.full(sim.all_t.shape, -50), "r--", label="Command bounds")
    ax2.plot(sim.all_t, np.full(sim.all_t.shape, 50), "r--")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position [m]")

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angle (servo) [deg]")

    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.show()
