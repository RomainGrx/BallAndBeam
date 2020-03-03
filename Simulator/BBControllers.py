# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 29-02-2020

import numpy as np
import scipy.optimize as opt
import abc
from Simulator import Simulator


class BBController(abc.ABC):
    """
    Classe abstraite qui englobe un objet 'BBSimulator' afin d'en permettre le controle. L'interface de controle
    utilisee se veut tres similaire a celle qui est mise a disposition pour le projet P4 MAP dans le programme
    LabVIEW. Cela est fait pour permettre un developpement du controleur qui reste proche de ce qui devra etre
    implemente a la fin du projet.
    """
    def __init__(self, sim):
        """
        Initialisation du controleur.

        :param sim : Objet de type 'BBSimulator' qui simule le systeme.
        """
        self.sim = sim
        self.dt = sim.dt
        self.buffer_size = sim.buffer_size
        self.flags = np.zeros((8,))         # Il y a huit 'flags' qui peuvent etre passes d'iteration en iteration.

    def control_law(self, ref, pos, dt, u_1, flags_1):
        """
        Loi de controle qui permet de calculer une nouvelle commande sur base de divers
        parametres. Les 'flags' peuvent etre modifies.

        Note: Dans LabVIEW, certaines valeurs sont passees en centimetres, mais ici on travaille en metres.
              Il suffira de faire la conversion au moment voulu. De meme, dans LabVIEW, on gere les angles
              en degres, mais ici on les gere en radians.

        :param ref     : Valeur de reference visee en cet instant [m].
        :param pos     : Position actuelle de la bille [m].
        :param dt      : Periode d'echantillonnage [s].
        :param u_1     : Commande au dernier appel du controleur [rad].
        :param flags_1 : Etat des 'flags' au dernier appel du controleur.
        :return        : 'u', la nouvelle commande a appliquer sur le systeme [rad].
        """
        
        raise NotImplementedError("BBController must implement method 'control_law'.")

    def simulate(self, setpoint, command_noise_func=lambda *args, **kwargs: 0,
                 output_noise_func=lambda *args, **kwargs: 0, n_steps=np.inf, init_state=0):
        """
        Fonction qui lance la simulation avec la loi de commande specifiee dans 'control_law'. Les valeurs
        de la reference visee sont contenues dans 'setpoint'. 'setpoint' doit contenir 'n_steps' elements
        ou plus.

        :param setpoint           : array de longueur 'n_steps' contenant ls valeurs du setpoint (la reference).
        :param command_noise_func : Fonction de bruit sur la commande (cf. docstring de 'Simulator.simulate').
        :param output_noise_func  : Fonction de bruit sur la sortie (cf. docstring de 'Simulator.simulate').
        :param n_steps            : Nombre de pas de temps a simuler.
        :param init_state         : Etat initial du systeme.
        :return                   : None
        """
        # Creation d'une 'command_func' qui se base sur la loi de controle specifiee dans 'control_law'.
        def command_func(timestep, params, all_t, all_u, all_y, dt):
            if timestep > 0:
                return self.control_law(setpoint[timestep], all_y[timestep - 1], dt, all_u[timestep - 1], self.flags)
            return self.control_law(setpoint[timestep], 0, dt, 0, self.flags)

        # On passe le tout a 'sim.simulate'.
        return self.sim.simulate(command_func, command_noise_func, output_noise_func, n_steps, init_state)


class PIDBBController(BBController):
    """
    Classe qui implemente une loi de controle de type PID pour le systeme Ball and Beam du projet P4 MAP.
    """
    def __init__(self, sim, kp, ki, kd):
        super().__init__(sim)
        self.kp, self.ki, self.kd = kp, ki, kd

    @Simulator.command_limiter(low_bound=np.deg2rad(-50), up_bound=np.deg2rad(50))
    def control_law(self, ref, pos, dt, u_1, flags_1):
        # Modifie self.flags et retourne u
        # Attention, dans LabVIEW ref est donne en cm, mais ici on le fait en m.
        # De meme, les angles sont geres en degres dans LabVIEW et en radians ici.
        #
        # Integrale de l'erreur avec le flag #0
        # Memorisation de la position precedente avec le flag #1
        vit=(pos-flags_1[1])/dt
        if(pos>0.35):
            ref=0.35
        if(pos<-0.35):
            ref=-0.35
        kp, ki, kd = self.kp, self.ki, self.kd          # A hardcoder dans LabVIEW
        theta_offset = self.sim.params["theta_offset"]  # A hardcoder dans LabVIEW
        err = pos - ref
        
        deriv_err = (pos - self.flags[1]) / dt
        self.flags[0] += err * dt
        self.flags[1] = pos
        integ_err = self.flags[0]
        return kp * err + ki * integ_err + kd * deriv_err - theta_offset


def fit_pid(sim, setpoint, init_values=None, method=None, bounds=None):
    """
    Fonction permettant de faire une minimisation de la MSE pour un controleur PID et sur un
    simulateur donne. Le simulateur est de type 'BBSimulator'. La minimisation se fait pour
    une suite de 'setpoints' donnee et n'a donc pas un caractere "general".

    :param sim         : Objet de type 'BBSimulator' qui gere la simulation du systeme.
    :param setpoint    : Array de points de reference sur lesquels la minimisation s'appuye.
    :param init_values : Valeurs initiales pour Kp, Ki et Kd. Prises au hasard si 'init_values'=None.
    :param method      : Methode a utiliser. Voir la documentation de 'scipy.optimize.minimize'.
    :param bounds      : Contraintes a utiliser sous forme d'une liste de paires (min, max).
                         Voir la documentation de 'scipy.optimize.minimize'.
    :return            : Un objet 'OptimizeResult' issu de la minimisation.
    """
    def err_func(params):
        # Fonction d'erreur calculant la MSE pour les parametres 'params'.
        kp, ki, kd = params
        cont = PIDBBController(sim, kp, ki, kd)
        cont.simulate(setpoint, n_steps=setpoint.size)
        mse = np.sum(np.power(np.absolute(setpoint - cont.sim.all_y[:n_steps].flatten()), 2)) / setpoint.size
        print("Kp = {:010.6f}; Ki = {:010.6f}; Kd = {:010.6f}    MSE = {:015.11f}".format(kp, ki, kd, mse))
        return mse

    if init_values is None:
        init_values = 5 * np.random.random(3)

    return opt.minimize(err_func, init_values, method=method, bounds=bounds)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as sig
    from BBSimulators import BBThetaSimulator

    sim = BBThetaSimulator()
    t = np.arange(0, 90, sim.dt)  # Simulation de 90s
    n_steps = t.size

    # setpoint = np.full(t.shape, 0.25)  # Setpoint constant: "maintenir la bille a une position fixee"
    setpoint = 0.3875 * np.sin(2 * np.pi * t/20)  # Setpoint = sinus de periode 9s et d'amplitude 0.15m
    #setpoint = 0.5 * sig.square(2 * np.pi * t / 9)  # Setpoint = carre de periode 9s et d'amplitude 0.15m

    # Decommenter les deux lignes ci-dessous pour lancer un fit du controleur PID sur la reference 'setpoint'
    # et pour le simulateur 'sim'
    # print(fit_pid(sim, setpoint, init_values=np.array([10.31712585,  0.49838698,  3.88553031]), method="Powell"))
    # exit()

    # Valeurs de parametres PID obtenues par optimisation sur un signal carre de periode 9s
    cont = PIDBBController(sim, 11.83864757, -0.05425518,  3.83534646)

    # Valeurs de parametres PID obtenues pour un setpoint constant a 0.25m
    # cont = PIDBBController(sim, 13.36836963,  0.22281434,  4.79696383)

    cont.simulate(setpoint, n_steps=n_steps)#,init_state=[0.15,0.0])
    
    fig, ((ax_pos), (ax_theta)) = plt.subplots(nrows=2, sharex=True)
    ax_pos.plot(t, setpoint, "ro--", linewidth=0.7, markersize=2, markevery=20, label="Setpoint [m]")
    ax_pos.plot(t, sim.all_y[:n_steps], "k-", label="Position [m]")
    ax_pos.plot(t, sim.all_y[:n_steps].flatten() - setpoint, "m--", linewidth=0.7, label="Error [m]")
    ax_theta.plot(t, np.rad2deg(sim.all_u[:n_steps] - sim.params["theta_offset"]), "b--",
                  linewidth=0.7, label="Commanded angle [deg]")
    ax_theta.plot(t, np.rad2deg(sim.all_u[:n_steps]), "g-", label="Actual offset angle [deg]")
    ax_theta.plot(t, np.full(t.shape, -50), "k--", linewidth=1, label="Commanded angle bounds [deg]")
    ax_theta.plot(t, np.full(t.shape, 50), "k--", linewidth=1)
    # ax_pos.legend()
    # ax_theta.legend()
    fig.legend(loc="right")
    ax_pos.grid()
    ax_theta.grid()
    ax_pos.set_xlabel("Time [s]")
    ax_pos.set_ylabel("Position [m]")
    ax_theta.set_xlabel("Time [s]")
    ax_theta.set_ylabel("Angle [deg]")
    plt.show()
