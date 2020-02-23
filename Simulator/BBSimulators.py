# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 22-02-2020

import numpy as np
import functools as ft
from Simulator import Simulator


class BBSimulator(Simulator):
    """
    Classe qui represente un simulateur "general" pour un systeme ball and beam decrit comme suit:
        - Vecteur d'etat      : [position, vitesse]
        - Vecteur de commande : [angle]  (cet angle est encore assez abstrait pour cette classe)
        - Vecteur de sortie   : [position]

    On assigne aussi des valeurs a plusieurs parametres caracteristiques du systeme Ball and Beam du P4 MAP.
    Note: La commande etant l'angle, on peut le faire varier de maniere instantanee avec ce simulateur.
          Cependant, dans un systeme reel, il y a un delai du a l'inertie de la poutre, notamment.
          Il faut garder cela en tete lors des tests et de la conception de la loi de commande.
    """
    def __init__(self, dt=0.05, buffer_size=100000):
        params = {
            "m": 55.9 / 1000,  # Masse de la bille [kg]
            "r": 30 / 1000,    # Rayon de la bille [m]
            "g": 9.81,         # Acceleration gravitationnelle a basse altitude [m/s^2]
            "rho": 1000,       # Masse volumique de l'eau [kg/m^3]
            "cx": 0.47,        # Coefficient de frottement d'une sphere []
            "l": 0.775,        # Longueur de la poutre [m]
            "d": 55 / 1000,    # Longueur de la premiere barre attachee au servo [m]
        }

        # Coefficient tel que le frottement fluide soit: Ff = -kf * v (approximation) []
        params["kf"] = 0.5 * params["rho"] * np.pi * params["r"]**2 * params["cx"]

        params["jb"] = 2/5 * params["m"] * params["r"]**2  # Moment d'inertie de la bille [kg*m^2]
        params["v"] = 4/3 * np.pi * params["r"]**3         # Volume de la bille [m^3]
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
    def __init__(self, dt=0.05, buffer_size=100000):
        super().__init__(dt, buffer_size)

    def dxdt(self):
        m, x, g, r = self.params["m"], self.all_x[self.timestep], self.params["g"], self.params["r"]
        jb, alpha, dalpha_dt = self.params["jb"], self.all_u[self.timestep], self.dudt()
        dx1_dt = x[1]
        dx2_dt = (m * x[0] * dalpha_dt**2 - m * g * np.sin(alpha)) / (jb / r**2 + m)
        return np.array([dx1_dt, dx2_dt])


class BBAlphaSimulator(BBSimulator):
    """
    Simulateur d'un ball and beam, dans lequel la commande correspond a l'angle de la poutre (alpha) et donc
    pas a l'angle du servo (theta). On tient compte des frottemtents fluides en les approximant par une fonction
    lineaire de la vitesse. On tient compte de la poussee d'Archimede. On tient compte des limites de la poutre
    (quand la balle arrive en bout de tube, elle bute). Le bridage de la commande ne depend pas de la classe mais
    de la fonction 'command_func' qui est passee en argument a 'simulate'.
    """
    def __init__(self, dt=0.05, buffer_size=100000):
        super().__init__(dt, buffer_size)

    def dxdt(self):
        m, x, g, r = self.params["m"], self.all_x[self.timestep], self.params["g"], self.params["r"]
        jb, alpha, dalpha_dt = self.params["jb"], self.all_u[self.timestep], self.dudt()
        rho, v, kf = self.params["rho"], self.params["v"], self.params["kf"]
        dx1_dt = x[1]
        dx2_dt = (m * x[0] * dalpha_dt**2 - m * g * np.sin(alpha) - (rho * v * g + kf) * x[1]) / (jb / r**2 + m)
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
                 output_noise_func=lambda *args, **kwargs: 0, n_steps=np.inf, init_state=0):
        # Ajout d'un mecanisme de verification de la validite de la position initiale.
        if np.size(init_state) == 1 and np.abs(init_state) > self.params["l"] / 2 or\
                np.size(init_state) == self.n_states and np.abs(init_state[0]) > self.params["l"] / 2:
            raise ValueError("Initial position is not on the beam")
        return super().simulate(command_func, command_noise_func, output_noise_func, n_steps, init_state)


class BBThetaSimulator(BBAlphaSimulator):
    """
    Meme chose que BBAlphaSimulator, mais ici c'est l'angle du servo (theta) qui sert de commande.
    """
    def dxdt(self):
        m, x, g, r = self.params["m"], self.all_x[self.timestep], self.params["g"], self.params["r"]
        d, l, jb = self.params["d"], self.params["l"], self.params["jb"]
        theta, dtheta_dt = self.all_u[self.timestep], self.dudt()
        alpha = np.arcsin(d / l * np.sin(theta))
        dalpha_dt = d * np.cos(theta) * self.dudt() / (l * np.sqrt(1 - (d * np.sin(theta) / l)**2))
        rho, v, kf = self.params["rho"], self.params["v"], self.params["kf"]
        dx1_dt = x[1]
        dx2_dt = (m * x[0] * dalpha_dt**2 - m * g * np.sin(alpha) - (rho * v * g + kf) * x[1]) / (jb / r**2 + m)
        return np.array([dx1_dt, dx2_dt])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as sig

    # Rappel important: ici, on a une commande "forte": l'angle est 100% commande
    # et si on le met a 0 et que la balle est initialement immobile, alors il ne
    # se passera rien de plus car l'angle sera maintenu a 0. Cela sera aussi le cas
    # si la balle est initialement decentree.
    # Cela signifie aussi que rien ne nous empeche de passer de (e.g.) +30deg a -30deg de maniere
    # instantanee, ce qui n'est pas realiste. Il faut en tenir compte lors des tests.

    # Note: Mettre un signal carre en entree engendrera une derivee de theta tres grande.
    # Puisque la vitesse depend de cette derivee, on observera un "catapultage" de la balle.
    # On observera ce catapultage uniquement si la balle est decentree, ce qui est coherent.

    def command_limiter(low_bound, up_bound):
        # Wrapper autour d'une fonction de commande pour en brider la sortie.
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

    # Fonction de commande dont la sortie est bridee entre -50deg et +50deg.
    # Decommenter les diverses lois de commande pour observer des comportements differents.
    @command_limiter(low_bound=np.deg2rad(-50), up_bound=np.deg2rad(50))
    def my_command_func(timestep, params, all_t, all_u, all_y, dt):
        # Donnees utilisables jusqu'a l'indice [timestep - 1]
        t = timestep * dt
        # return 0
        return np.deg2rad(50)
        # return np.array([-np.pi / 6])
        # return np.sin(t * 1) * (-np.pi / 2)
        # return np.sin(t * 1) * np.deg2rad(-50)
        # return all_u[timestep] + (np.random.random(1) * 2 - 1) * 0.05
        # return np.sin(t * 10) * (-np.pi / 6)
        # return np.sin(t * 1) * (-np.pi / 6) * t
        # return sig.square(t * 1, duty=0.5) * (-np.pi / 12)
        # return sig.square(t * 1, duty=0.5) * (-np.pi / 12) + np.pi / 12
        # return = (all_y[timestep - 1] - 0.42) * 2

    # Fonction de bruit aleatoire (distribution uniforme) sur la commande.
    def my_command_noise_func(timestep, params, all_t, all_u, all_y, dt):
        return (2 * np.random.random(1) - 1) * np.deg2rad(3)  # Erreur de +- 3deg a la commande du servo

    # Fonction de bruit aleatoire (distribution uniforme) sur la mesure de la position.
    def my_output_noise_func(timestep, params, all_t, all_u, all_y, dt):
        return (2 * np.random.random(1) - 1) * 0.025  # Erreur de +- 2.5cm a la mesure

    # sim = BBSimpleSimulator(dt=0.05, buffer_size=1000)
    # sim = BBAlphaSimulator(dt=0.05, buffer_size=1000)
    sim = BBThetaSimulator(dt=0.05, buffer_size=1000)

    my_init_state = np.array([0, 0])
    # my_init_state = np.array([0, -0.333])
    # my_init_state = np.array([1, 0])

    sim.simulate(my_command_func, my_command_noise_func, my_output_noise_func, init_state=my_init_state)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True)

    ax1.plot(sim.all_t, sim.all_y, "k-", linewidth=2, label="Simulated position")
    ax1.plot(sim.all_t, np.full(sim.all_t.shape, -sim.params["l"] / 2), "r--", label="Position bounds")
    ax1.plot(sim.all_t, np.full(sim.all_t.shape, sim.params["l"] / 2), "r--")

    ax2.plot(sim.all_t, np.rad2deg(sim.all_u), "k-", linewidth=2, label="Command")
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
