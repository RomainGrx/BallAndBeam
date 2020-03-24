# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 24-03-2020

import BBSimulators as sim
import BBControllers as cont


def idiot_proof_test(x_init, u, flag_idiot_proof):
    """
    :param x_init           : condition initiale ([cm] avec point de reference au milieu du “beam”)
    :param u                : le vecteur de commande du moteur [deg]
    :param flag_idiot_proof : 1 si le mecanisme d'idiot-proof est actif, 0 sinon
    :return                 : y [cm], l'evolution de l'etat du systeme
    """
    kp, ki, kd = 1, 2, 3
    s = sim.BBThetaSimulator()
    c = cont.Obj3PIDBBController
    return y