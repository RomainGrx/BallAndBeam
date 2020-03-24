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
    kp, ki, kd = 5.13051124e+01, -1.59963530e-02,  9.82885344e+00
    s = sim.BBThetaSimulator()
    c = cont.Obj3PIDBBController(s, kp, ki, kd, bool(flag_idiot_proof))

    # Conversion des unites (cm -> m; deg -> rad)
    x_init /= 100
    # TODO: attention, peut-etre que u est une bete liste, il faut s'assurer d'en faire un ndarray.

    # TODO: debut d'implementation dans FreeControl (a supprimer si inutilise)
    # TODO: cf. reponse a notre question sur le forum.

    # TODO: conversion unites!
    return y

def multiple_positions(x_init, x_desired):
    """
    :param x_init    : position initiale de la bille [cm]
    :param x_desired : sequence de positions desirees: le but est de stabiliser une seconde sur chacune d'elles.
    :return          : 'y' est l'evolution de l'etat du systeme (position de la bille [cm]).
    """

    # TODO: x_init est juste une position, PAS une vitesse initiale!
    # TODO: transformer la "sequence de positions a tenir 1s chacune" en un vrai setpoint (1s = 20 * 0.05s)
    # TODO: Attention aux conversions d'unites

    # Setup du simulateur et du controleur
    kp, ki, kd = 5.13051124e+01, -1.59963530e-02, 9.82885344e+00
    s = sim.BBThetaSimulator()
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

    # Test de la fonction d'interface 'multiple_positions'
    x_init = 0
    x_desired = np.repeat(np.arange(-45, 45 + 1, 5), 5)
    y = multiple_positions(x_init, x_desired)
    t = np.arange(0, x_desired.size, 0.05)
    plt.plot(t, y, t, np.repeat(x_desired, 20))
    plt.grid()
    plt.show()
