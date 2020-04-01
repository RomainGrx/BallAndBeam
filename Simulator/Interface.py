# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 24-03-2020


import numpy as np

import BBSimulators as sim
import BBControllers as cont


def idiot_proof_test(x_init, u, flag_idiot_proof):
    """
    :param x_init           : condition initiale ([cm] avec point de reference au milieu du “beam”)
    :param u                : le vecteur de commande du moteur [deg]
    :param flag_idiot_proof : 1 si le mecanisme d'idiot-proof est actif, 0 sinon
    :return                 : y [cm], l'evolution de l'etat du systeme
    """
    # Setup du simulateur et du controleur
    s = sim.BBThetaSimulator(buffer_size=u.size + 1)
    c = cont.FreeControlBBController(s, bool(flag_idiot_proof))

    # Conversion des unites (cm -> m; deg -> rad)
    x_init /= 100
    u = np.deg2rad(u)

    # Lancement de la simulation
    c.simulate(u, n_steps=u.size, init_state=np.array([x_init, 0]))

    return s.all_y[:u.size] * 100


def multiple_positions(x_init, x_desired):
    """
    :param x_init    : position initiale de la bille [cm]
    :param x_desired : sequence de positions desirees: le but est de stabiliser une seconde sur chacune d'elles.
    :return          : 'y' est l'evolution de l'etat du systeme (position de la bille [cm]).
    """

    # Setup du simulateur et du controleur
    kp, ki, kd = 5.13051124e+01, -1.59963530e-02, 9.82885344e+00
    s = sim.BBThetaSimulator(buffer_size=x_desired.size * 20 + 1)
    c = cont.Obj3PIDBBController(s, kp, ki, kd, using_idiot_proofing=True)

    # Conversion des unites (cm -> m; deg -> rad)
    x_init /= 100
    x_desired = np.array(x_desired) / 100  # Au cas ou x_desired ne serait qu'une liste.

    # Transformation de x_desired en une vraie trajectoire (1 point par 0.05s -> 20 points par seconde)
    x_desired = np.repeat(x_desired, 20)

    # Lancement de la simulation
    n_steps = x_desired.size
    c.simulate(x_desired, n_steps=n_steps, init_state=np.array([x_init, 0]))

    # Recuperation de la sortie et conversion m -> cm
    y = s.all_y[:n_steps] * 100

    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as sig

    test_function = "idiot_proof_test"
    # test_function = "multiple_positions"

    if test_function == "idiot_proof_test":
        # Test de la fonction d'interface 'idiot_proof_test'
        x_init = 10
        t = np.arange(0, 60, 0.05)
        u = 40 * sig.square(2 * np.pi * t / 15, 0.5)
        y = idiot_proof_test(x_init, u, flag_idiot_proof=True)
        plt.plot(t, y, t, u)
        plt.grid()
        plt.show()

    elif test_function == "multiple_positions":
        # Test de la fonction d'interface 'multiple_positions'
        x_init = 0
        x_desired = np.repeat(np.arange(-45, 45 + 1, 5), 5)
        y = multiple_positions(x_init, x_desired)
        t = np.arange(0, x_desired.size, 0.05)
        plt.plot(t, y, t, np.repeat(x_desired, 20))
        plt.grid()
        plt.show()
