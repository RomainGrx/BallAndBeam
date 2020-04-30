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
    kp, ki, kd = 1.88700101e+01, -1.93159758e-03, 7.47877146e+00
    s = sim.BBThetaSimulator(buffer_size=x_desired.size * 20 + 1)
    c = cont.Obj5PIDBBController(s, kp, ki, kd, using_idiot_proofing=True)

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


def multiple_positions_avec_contraintes(x_init, x_1, x_2, v_max, v_min, perturbation=lambda *args: args[0]):
    """
    :param x_init       : Position initiale de la bille [cm]
    :param x_1          : Position de debut de la trajectoire a effectuer [cm]
    :param x_2          : Position de fin de la trajectoire a effectuer [cm]
    :param v_max        : Vitesse maximale sur la trajectoire [cm/s]
    :param v_min        : Vitesse minimale sur la trajectoire [cm/s]
    :param perturbation : Fonction pour perturber l'acceleration: 'perturbation(a_desired, x, v, alpha, t)'
    :return             : Evolution de l'etat du systeme (position de la bille [cm])
    """
    # Conversion des unites
    x_init, x_1, x_2, v_min, v_max = x_init / 100, x_1 / 100, x_2 / 100, v_min / 100, v_max / 100

    def si_perturbation(a_desired, x, v, alpha, t):
        return 0.01 * perturbation(100 * a_desired, 100 * x, 100 * v, np.rad2deg(alpha), t)

    kp, ki, kd = 1.88700101e+01, -1.93159758e-03, 7.47877146e+00
    # Setup du simulateur et du
    t = np.arange(0, 125, 0.05)  # 25s a semble suffisant pour n'importe quelle trajectoire (on met une marge de x5)
    n_steps = t.size
    s = sim.BBObj7Simulator(si_perturbation, buffer_size=n_steps + 1)
    c = cont.Obj7Controller(s,kp, ki, kd, x_1, x_2, v_min, v_max, using_idiot_proofing=True)

    # Lancement de la simulation
    c.simulate(np.empty(t.shape), n_steps=n_steps, init_state=np.array([x_init, 0]))

    # Recuperation de la sortie et conversion m -> cm
    y = s.all_y[:n_steps] * 100

    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as sig

    # Pour tester l'objectif #3
    # test_function = "idiot_proof_test"

    # Pour tester l'objectif #4 ou #5
    # test_function = "multiple_positions"

    # Pour tester l'objectif #6 ou l'objectif #7 (dans ce cas, utiliser la fonction 'perturbation' ci-dessous)
    test_function = "multiple_positions_avec_contraintes"

    # Pour tester l'objectif #7
    def perturbation(a_desired, x, v, alpha, t):
        # Unites: [cm/s^2], [cm], [cm/s], [deg], [s]; Retour: [cm/s^2]
        return a_desired

    if test_function == "idiot_proof_test":
        # Test de la fonction d'interface 'idiot_proof_test'
        x_init = 10
        t = np.arange(0, 60, 0.05)
        u = 40 * sig.square(2 * np.pi * t / 15, 0.5)
        y = idiot_proof_test(x_init, u, flag_idiot_proof=True)
        plt.plot(t, y, label="Position [cm]")
        plt.plot(t, u, label="Commanded angle (servo) [deg]")

    elif test_function == "multiple_positions":
        # Test de la fonction d'interface 'multiple_positions'
        x_init = 0
        x_desired = np.repeat(np.arange(-45, 45 + 1, 5), 5)
        y = multiple_positions(x_init, x_desired)
        t = np.arange(0, x_desired.size, 0.05)
        plt.plot(t, y, label="Position [cm]")
        plt.plot(t, np.repeat(x_desired, 20), label="Setpoint [cm]")

    elif test_function == "multiple_positions_avec_contraintes":
        # Test de la fonction d'interface 'multiple_positions_avec_contraintes'
        x_init = -10
        x_1, x_2, v_min, v_max = 20, -5, 3, 4
        y = multiple_positions_avec_contraintes(x_init, x_1, x_2, v_max, v_min, perturbation)
        t = np.arange(0, 125, 0.05)
        plt.plot(t, y, label="Position [cm]")
        plt.plot(t, np.full(t.shape, x_1), "r--", label="$x_1$ and $x_2$")
        plt.plot(t, np.full(t.shape, x_2), "r--")

    plt.xlabel("Time [s]")
    plt.ylabel("Position [cm] / Angle [deg]")
    plt.legend()
    plt.grid()
    plt.show()
