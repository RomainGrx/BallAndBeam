# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sig
from Simulator import Simulator
import functools as ft
from BBSimulators import BBThetaSimulator
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Rappel important: ici, on a une commande "forte": l'angle est 100% commande
    # et si on le met a 0 et que la balle est initialement immobile, alors il ne
    # se passera rien de plus car l'angle sera maintenu a 0. Cela sera aussi le cas
    # si la balle est initialement decentree.
    # Cela signifie aussi que rien ne nous empeche de passer de (e.g.) +30deg a -30deg de maniere
    # instantanee, ce qui n'est pas realiste. Il faut en tenir compte lors des tests.

    # Note: Mettre un signal carre en entree engendrera une derivee de theta tres grande.
    # Puisque la vitesse depend de cette derivee, on observera un "catapultage" de la balle.
    # On observera ce catapultage uniquement si la balle est decentree, ce qui est coherent.

    @Simulator.command_limiter(low_bound=np.deg2rad(-50), up_bound=np.deg2rad(50))
    def prop_command(timestep, params, all_t, all_u, all_y, dt):
        setpoint_func = lambda timestep: 0.25 * sig.square(timestep * dt * 2 * np.pi / 20, duty=0.5)
        err_func = lambda timestep: all_y[timestep - 1] - setpoint_func(timestep)
        integ_func = lambda timestep: (all_y[:timestep - 1] - setpoint_func(np.arange(0, timestep)))
        kp = 100
        kd = 0
        kp = 0

        err = err_func(timestep)
        if timestep >= 1:
            derrdt = (err_func(timestep) - err_func(timestep - 1)) / dt
        else:
            derrdt = 0

        return kp * err + kd * derrdt

    # Fonction de bruit aleatoire (distribution uniforme) sur la commande.
    def my_command_noise_func(timestep, params, all_t, all_u, all_y, dt):
        return 0 #(2 * np.random.random(1) - 1) * np.deg2rad(3)  # Erreur de +- 3deg a la commande du servo

    # Fonction de bruit aleatoire (distribution uniforme) sur la mesure de la position.
    def my_output_noise_func(timestep, params, all_t, all_u, all_y, dt):
        return 0 #(2 * np.random.random(1) - 1) * 0.025  # Erreur de +- 2.5cm a la mesure

    sim = BBThetaSimulator(dt=0.05, buffer_size=1000)

    my_init_state = np.array([0, 0])

    sim.simulate(prop_command, my_command_noise_func, my_output_noise_func, init_state=my_init_state)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True)

    ax1.plot(sim.all_t, sim.all_y, "k-", linewidth=2, label="Simulated position")
    ax1.plot(sim.all_t, np.full(sim.all_t.shape, -sim.params["l"] / 2), "r--", label="Position bounds")
    ax1.plot(sim.all_t, np.full(sim.all_t.shape, sim.params["l"] / 2), "r--")
    ax1.plot(sim.all_t, 0.25 * sig.square(sim.all_t * 2 * np.pi / 20, duty=0.5), label="Setpoint")

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
