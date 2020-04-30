# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 22-02-2020

import functools as ft
import numpy as np
import abc


class Simulator(abc.ABC):
    """
    Classe abstraite qui gere une simulation d'un systeme non-lineaire en temps discret (periode 'dt'):
        - Vecteur d'etat                    : x
        - Vecteur de commande               : u
        - Vecteur de sortie                 : y
        - Bruit sur la commande             : vu
        - Bruit sur la sortie               : vy
        - Fonction derivee de l'etat        : f(x, u, vu)
        - Fonction descriptive de la sortie : h(x, u, yu)

    Dynamique du systeme:
    d/dt x(t) = f(x, u, vu)
    y(t)      = h(x, u, yu)

    La valeur des differentes variables du systeme sont enregistrees pour un maximum de 'buffer_size' iterations.
    Cette classe n'est pas faite pour etre instanciee. Il faut en faire de l'heritage et implementer les
    methodes non implementees en fonction du modele que l'on veut simuler.
    """

    @staticmethod
    def command_limiter(low_bound, up_bound):
        """
        Decorateur pour des fonctions de commande. Il s'agit d'un limiteur qui permet de brider la sortie
        de la fonction de commande decoree entre 'low_bound' et 'up_bound'. Ce decorateur ne fonctionne que
        pour une commande scalaire (a.k.a. pas un vecteur).

        :param low_bound : Borne reelle inferieure pour la commande.
        :param up_bound  : Borne reelle superieure pour la commande.
        :return          : Fonction de commande decoree bridee.
        """
        def decorator(command_func):
            @ft.wraps(command_func)
            def wrapper(*args, **kwargs):
                raw_command = command_func(*args, **kwargs)
                if raw_command < low_bound:
                    return low_bound
                elif raw_command > up_bound:
                    return up_bound
                else:
                    return raw_command
            return wrapper
        return decorator

    def __init__(self, params, n_states, n_commands, n_outputs, dt=0.05, buffer_size=10000):
        """
        Initialisation du simulateur.

        :param params      : Dictionnaire de parametres physiques descriptifs du systeme.
        :param n_states    : Nombre d'etats du systeme.
        :param n_commands  : Nombre de commandes du systeme.
        :param n_outputs   : Nombre de sorties du systeme.
        :param dt          : Periode d'echantillonnage pour la discretisation du systeme [s].
        :param buffer_size : Nombre maximal de donnees enregistrables par le systeme.
        """

        # Numero du pas de temps courant. Utilise lors de la simulation. Le temps est donne par 'dt' * 'timestep'.
        self.timestep = 0

        # Enregistrement des divers attributs de l'objet.
        self.params = params
        self.dt = dt
        self.buffer_size = buffer_size
        self.n_states = n_states
        self.n_commands = n_commands
        self.n_outputs = n_outputs

        # Creation de vecteurs de sauvegarde pour les donnees. Ces vecteurs ne peuvent etre remplis au-dela
        # de 'buffer_size' elements. Si cela arrive, l'arret de la simulation sera force.
        # Notes importantes:
        #   - 'all_u' et 'all_y' tiennent compte du bruit (toute interaction avec le systeme est bruitee);
        #   - 'all_x' n'est pas bruite, car il s'agit d'un etat "theorique" du systeme.
        self.all_t = np.arange(0, self.buffer_size * self.dt, self.dt)
        self.all_x = np.zeros((self.buffer_size, self.n_states))
        self.all_u = np.zeros((self.buffer_size, self.n_commands))
        self.all_u_noise = np.zeros((self.buffer_size, self.n_commands))
        self.all_y = np.zeros((self.buffer_size, self.n_outputs))
        self.all_y_noise = np.zeros((self.buffer_size, self.n_outputs))

    @abc.abstractmethod
    def dxdt(self):
        """
        Implementation de la fonction 'f(x, u, vu)' pour calculer la derivee de l'etat au temps 'timestep' * 'dt'.
        Il ne faut pas inclure de bruit dans cette fonction car on est dans la partie "theorique" du systeme, alors
        que le bruit n'intervient que lors de la partie "pratique": les interactions avec le systeme.

        Note: Lors de l'appel de cette fonction, le vecteur 'all_x' aura ete rempli jusqu'a l'indice
              'timestep' inclus.

        :return: Vecteur contenant la derivee de l'etat au temps 'timestep'
        """
        # En l'occurence on est dans la classe abstraite. Les implementations se feront dans les sous-classes.
        raise NotImplementedError("Method 'dxdt' in class 'Simulator' must be overridden.")

    @abc.abstractmethod
    def y(self):
        """
        Implementation de la fonction 'h(x, u, yu)' pour calculer la sortie du systeme au temps 'timestep' * 'dt'.
        Cette fonction ne tient pas compte du bruit, car celui-ci est gere dans 'simulate' avec 'output_noise_func'.

        :return: Vecteur contenant la sortie non-bruitee au temps 'timestep' * 'dt' (sortie theorique).
        """
        # En l'occurence on est dans la classe abstraite. Les implementations se feront dans les sous-classes.
        raise NotImplementedError("Method 'y' in class 'Simulator' must be overridden.")

    def dudt(self):
        """
        Methode permettant de calculer la derivee de la commande au temps 'timestep' * 'dt' sur base de
        methodes de differences finies arrieres.

        Voir: https://en.wikipedia.org/wiki/Finite_difference_coefficient#Backward_finite_difference

        :return: Vecteur contenant une approximation de la derivee de la commande au temps 'timestep' * 'dt'.
        """
        if self.timestep == 0:
            # Approximation triviale (manque de donnees)
            return 0
        elif self.timestep == 1:
            # Euler explicite (difference finie arriere d'ordre 1)
            return (self.all_u[self.timestep] - self.all_u[self.timestep - 1]) / self.dt
        elif self.timestep == 2:
            # Difference finie arriere d'ordre 2
            return (3/2 * self.all_u[self.timestep] - 2 * self.all_u[self.timestep - 1] +
                    1/2 * self.all_u[self.timestep - 2]) / self.dt
        else:
            # Difference finie arriere d'ordre 3
            return (11/6 * self.all_u[self.timestep] - 3 * self.all_u[self.timestep - 1] +
                    3/2 * self.all_u[self.timestep - 2] - 1/3 * self.all_u[self.timestep - 3]) / self.dt

    def update_state(self):
        """
        Fonction qui procede a la mise a jour de l'etat du systeme pour le temps ('timestep' + 1) * 'dt'.
        C'est donc une mise a jour pour l'iteration suivante. La mise a jour s'effectue avec la methode
        d'Euler explicite.

        Voir: https://en.wikipedia.org/wiki/Finite_difference_coefficient#Backward_finite_difference

        :return: None
        """
        dt, dxdt = self.dt, self.dxdt()
        self.all_x[self.timestep + 1] = self.all_x[self.timestep] + dt * dxdt

    def simulate(self, command_func, command_noise_func=lambda *args, **kwargs: 0,
                 output_noise_func=lambda *args, **kwargs: 0, n_steps=np.inf, init_state=0):
        """
        Methode qui effectue une simulation sur 'n_steps' pas de temps, sauf si cette valeur depasse
        'buffer_size', auquel cas la simulation s'arrete lorsque les buffers sont remplis.
        Cette simulation tient compte d'un bruit sur la commande et d'un autre sur la sortie.

        :param command_func       : Fonction de retournant la commande au temps 'timestep'. Signature:
                                    command_func(timestep, params, all_t, all_u, all_y, dt)
        :param command_noise_func : Fonction retournant le bruit a ajouter a la commande au temps 'timestep'. Signature:
                                    command_noise_func(timestep, params, all_t, all_u, all_y, dt)
        :param output_noise_func  : Fonction retournant le bruit a ajouter a la sortie au temps 'timestep'. Signature:
                                    output_noise_func(timestep, params, all_t, all_u, all_y, dt)
        :param n_steps            : Nombre de pas de temps desires. Le nombre effectif de pas de temps est donne par:
                                    min(n_steps, buffer_size)
        :param init_state         : Etat initial du systeme sous forme d'un array comportant 'n_states' elements.
        :return                   : None
        """
        # Initialisation du systeme
        if np.size(init_state) == 1 and self.n_states != 1:
            init_state = np.full((self.n_states,), init_state)

        self.timestep = 0
        self.all_x = np.zeros((self.buffer_size, self.n_states))
        self.all_u = np.zeros((self.buffer_size, self.n_commands))
        self.all_y = np.zeros((self.buffer_size, self.n_outputs))
        self.all_x[0] = init_state

        # Boucle principale de la simulation
        while self.timestep < min(self.buffer_size, n_steps):
            # 1) Calcul de la commande au temps 'timestep' sur base de l'etat du systeme au temps 'timestep'
            u = command_func(self.timestep, self.params, self.all_t, self.all_u, self.all_y, self.dt)
            u_noise = command_noise_func(self.timestep, self.params, self.all_t, self.all_u, self.all_y, self.dt)
            self.all_u[self.timestep] = u + u_noise
            self.all_u_noise[self.timestep] = u_noise

            # 2) Mise a jour de l'etat du systeme au temps 'timestep' + 1 sur base du systeme au temps 'timestep'
            if self.timestep + 1 < self.buffer_size:
                self.update_state()

            # 3) Calcul de la sortie du systeme au temps 'timestep' sur base du systeme au temps 'timestep'
            y = self.y()
            y_noise = output_noise_func(self.timestep, self.params, self.all_t, self.all_u, self.all_y, self.dt)
            self.all_y[self.timestep] = y + y_noise
            self.all_y_noise[self.timestep] = y_noise

            self.timestep += 1

        # Gestion du cas ou le buffer est rempli avant la fin de la simulation
        if self.timestep == self.buffer_size and self.buffer_size <= n_steps < np.inf:
            raise RuntimeWarning("Simulation stopped as the simulation buffer is full.")


# La suite du code ne sera executee que si le fichier 'Simulator.py' est lance directement. Elle ne le sera pas si ce
# fichier est utilise comme import dans un autre fichier. La section ci-dessous sert de code de demonstration pour
# l'utilisation des classes implementees ci-dessus.
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Exemple avec le systeme y = x et dx/dt = -x
    # Pour modeliser ce systeme, on herite de la classe 'Simulator' et on implemente les methodes abstraites
    class ExpSimulator(Simulator):
        def __init__(self, dt=0.05, buffer_size=10000):
            n_states, n_commands, n_outputs = 1, 0, 1
            super().__init__(dict(), n_states, n_commands, n_outputs, dt, buffer_size)

        def dxdt(self):
            return -self.all_x[self.timestep]

        def y(self):
            return self.all_x[self.timestep]

    # Implementation de fonctions de bruit sur la commande et sur la sortie. En l'occurence, on ne met pas de bruit sur
    # la commande mais on met un bruit sur la sortie.
    def my_command_func(timestep, params, all_t, all_u, all_y, dt):
        return np.zeros((0,))

    def my_output_noise(timestep, params, all_t, all_u, all_y, dt):
        # Erreur uniformement repartie entre -2.5cm et 2.5cm sur la position lors de la mesure
        return (np.random.random(1) * 2 - 1) * 0.025


    # Pour pouvoir estre utilise, le simulateur doit etre instancie. On specifie 'buffer_size=1000' et 'dt=0.01',
    # c'est-a-dire que les simulations auront une duree maximale de 10s (1000 * 0.01 = 10)
    sim = ExpSimulator(dt=0.01, buffer_size=1000)

    # On definit egalement l'etat initil du systeme. Ici, on choisit x = 0m
    my_init_state = np.array([1])

    # On lance la simulation avec la methode 'simulate' de notre simulateur, en specifiant les bruits et l'etat initial
    sim.simulate(my_command_func, output_noise_func=my_output_noise, init_state=my_init_state)

    # On termine en recuperant les donnees de la simulation afin de les afficher dans un graphe
    # Dans ce graphe, on affiche la position simulee non-bruitee ('Actual position'), la position mesuree en incluant
    # le bruit ('Measured position') ainsi que la solution analytique du modele mathematique ('f(t) = exp(-t)').
    plt.plot(sim.all_t, sim.all_y, "k-", linewidth=1, label="Measured position [m]")
    plt.plot(sim.all_t, sim.all_x, "b-", linewidth=2, label="Actual position [m]")
    plt.plot(sim.all_t, np.exp(-sim.all_t), "r--", linewidth=1, label="f(t) = exp(-t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid()
    plt.legend()
    plt.show()
