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
        self.flags = np.zeros((8,))  # Il y a huit 'flags' qui peuvent etre passes d'iteration en iteration.

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
        # Memorisation de l'erreur precedente avec le flag #1

        self.flags = flags_1  # Necessaire pour MathScript (sinon LabVIEW crash)
        # ref = pos

        kp, ki, kd = self.kp, self.ki, self.kd  # A hardcoder dans LabVIEW
        theta_offset = self.sim.params["theta_offset"]  # A hardcoder dans LabVIEW
        err = pos - ref

        deriv_err = (err - self.flags[1]) / dt
        self.flags[0] += err * dt
        self.flags[1] = err
        integ_err = self.flags[0]

        return kp * err + ki * integ_err + kd * deriv_err - theta_offset


class Obj3PIDBBController(BBController):
    """
    Classe qui implemente une loi de controle de type PID pour le systeme Ball and Beam du projet P4 MAP.
    Ce controleur implemente aussi une version du "idiot proofing".
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
        # Memorisation de l'erreur precedente avec le flag #2

        # Pour l'objectif 3, on a besoin de donner une petite vitesse a la balle
        # on utilise les flags 0 et 1 pour bypasser le controller sur les quelques premieres
        # iterations.
        self.flags = flags_1
        if self.flags[0] == 0:
            self.flags[0] = 1
            self.flags[1] = 0#200  # On bypass le controle sur 200 iterations (10s)

        if self.flags[0] == 1 and self.flags[1] > 0:
            # Phase "d'initialisation"
            self.flags[1] -= 1
            return np.deg2rad(-30)
        else:
            # Phase "controleur"
            # Parametres du PID:
            kp, ki, kd = self.kp, self.ki, self.kd  # A hardcoder dans LabVIEW
            theta_offset = self.sim.params["theta_offset"]  # A hardcoder dans LabVIEW

            # Idiot proofing:
            # Entre a_pos et b_pos, on applique un angle qui varie lineairement entre alpha_a_pos et alpha_b_pos
            # pour forcer une correction de la trajectoire. Le controle et ref sont bypasses dans ce cas.
            a_pos = 0.35
            b_pos = 0.775 / 2
            alpha_a_pos, alpha_b_pos = np.deg2rad(20), np.deg2rad(49)
            if pos > a_pos:
                return alpha_a_pos + (pos - a_pos) * (alpha_b_pos - alpha_a_pos) / (b_pos - a_pos) - theta_offset
            elif pos < -a_pos:
                return -alpha_a_pos + (pos + a_pos) * (alpha_b_pos - alpha_a_pos) / (b_pos - a_pos) - theta_offset

            # Si la reference est abberrante (i.e.: hors limites), on la bride une premiere fois.
            # Faire ceci permet derelaxer un peu le reste de l'idiot-proofing afin de pouvoir s'approcher un peu
            # plus des bords pour les trajectoires "normales" (i.e. pas hors limites).
            if ref > 0.775 / 2:
                ref = 0.775 / 2
            if ref < -0.775 / 2:
                ref = -0.775 / 2

            # Si il n'y a pas lieu de faire une correction sur la position, mais qu'on voit que ref s'approche un peu
            # trop violemment des limites a_pos et b_pos sur la position, alors on applique une correction de type
            # "1/x" sur ref pour que la balle s'approche plus doucement des limites de position.
            # La fonction de correction est telle que:
            # f(a_ref) = a_ref; f(+inf) = b_ref; evolution de type 1/x entre les deux.
            a_ref = 0.80 * a_pos
            b_ref = 0.90 * a_pos
            if ref > a_ref:
                ref = b_ref - a_ref * (b_ref - a_ref) / ref
            if ref < -a_ref:
                ref = -b_ref - a_ref * (b_ref - a_ref) / ref
            # Fin de l'idiot-proofing

            # Controle PID
            err = pos - ref

            deriv_err = (err - self.flags[2]) / dt
            self.flags[0] += err * dt
            self.flags[2] = err
            integ_err = self.flags[0]

            return kp * err + ki * integ_err + kd * deriv_err - theta_offset


def fit_pid(sim, setpoint_list, init_values=None, method=None, bounds=None):
    """
    Fonction permettant de faire une minimisation de l'erreur pour un controleur PID et sur un
    simulateur donne. Le simulateur est de type 'BBSimulator'. La minimisation se fait pour
    une suite de 'setpoints' donnee et n'a donc pas un caractere "general".
    :param sim           : Objet de type 'BBSimulator' qui gere la simulation du systeme.
    :param setpoint_list : Liste d'array de points de reference (setpoints) sur lesquels la minimisation s'appuye.
    :param init_values   : Valeurs initiales pour Kp, Ki et Kd. Prises au hasard si 'init_values'=None.
    :param method        : Methode a utiliser. Voir la documentation de 'scipy.optimize.minimize'.
    :param bounds        : Contraintes a utiliser sous forme d'une liste de paires (min, max).
                         Voir la documentation de 'scipy.optimize.minimize'.
    :return              : Un objet 'OptimizeResult' issu de la minimisation.
    """

    def err_func(params):
        # Fonction d'erreur calculant l'erreur pour les parametres 'params'.
        kp, ki, kd = params
        cont = PIDBBController(sim, kp, ki, kd)
        tot_err = 0
        for setpoint in setpoint_list:
            cont.simulate(setpoint, n_steps=setpoint.size)
            tot_err += np.sum(np.absolute(setpoint - cont.sim.all_y[:n_steps].flatten())) / setpoint.size
        print("Kp = {:010.6f}; Ki = {:010.6f}; Kd = {:010.6f}    mean error per setpoint = {:015.11f}"
              "".format(kp, ki, kd, tot_err / len(setpoint_list)))
        return tot_err

    if init_values is None:
        init_values = 5 * np.random.random(3)

    return opt.minimize(err_func, init_values, method=method, bounds=bounds)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as sig
    from BBSimulators import BBThetaSimulator

    sim = BBThetaSimulator()
    t = np.arange(0, 120, sim.dt)  # Simulation d'un certain nombre de secondes (2e argument)
    n_steps = t.size

    # Liste des setpoints sur lesquels on se base pour fit le PID.
    # Le fit s'effectue sur un controleur depourvu d'idiot-proofing.
    setpoints_to_fit = (
        # Quelques trajectoires constantes:
        np.zeros(t.shape),
        # np.full(t.shape, -0.30),
        np.full(t.shape, -0.20),
        # np.full(t.shape, -0.10),
        # np.full(t.shape, 0.10),
        np.full(t.shape, 0.20),
        # np.full(t.shape, 0.30),
        # Quelques trajectoires sinusoidales:
        # 0.1 * np.sin(2 * np.pi * t / 30),
        # 0.2 * np.sin(2 * np.pi * t / 30),
        # 0.3 * np.sin(2 * np.pi * t / 30),
        # -0.1 * np.sin(2 * np.pi * t / 30),
        # -0.2 * np.sin(2 * np.pi * t / 30),
        # -0.3 * np.sin(2 * np.pi * t / 30),
        # 0.1 * np.sin(2 * np.pi * t / 15),
        0.2 * np.sin(2 * np.pi * t / 15),
        # 0.3 * np.sin(2 * np.pi * t / 15),
        -0.1 * np.sin(2 * np.pi * t / 15),
        # -0.2 * np.sin(2 * np.pi * t / 15),
        # -0.3 * np.sin(2 * np.pi * t / 15),
        0.1 * np.sin(2 * np.pi * t / 9),
        # 0.2 * np.sin(2 * np.pi * t / 9),
        # 0.3 * np.sin(2 * np.pi * t / 9),
        # -0.1 * np.sin(2 * np.pi * t / 9),
        -0.2 * np.sin(2 * np.pi * t / 9),
        # -0.3 * np.sin(2 * np.pi * t / 9),
        # 0.1 * np.sin(2 * np.pi * t / 7),
        # 0.2 * np.sin(2 * np.pi * t / 7),
        # 0.3 * np.sin(2 * np.pi * t / 7),
        # -0.1 * np.sin(2 * np.pi * t / 7),
        # -0.2 * np.sin(2 * np.pi * t / 7),
        # -0.3 * np.sin(2 * np.pi * t / 7),
        # Quelques trajectoires carrees:
        # 0.1 * sig.square(2 * np.pi * t / 30),
        # 0.2 * sig.square(2 * np.pi * t / 30),
        # 0.3 * sig.square(2 * np.pi * t / 30),
        # -0.1 * sig.square(2 * np.pi * t / 30),
        # -0.2 * sig.square(2 * np.pi * t / 30),
        # -0.3 * sig.square(2 * np.pi * t / 30),
        # 0.1 * sig.square(2 * np.pi * t / 15),
        0.2 * sig.square(2 * np.pi * t / 15),
        # 0.3 * sig.square(2 * np.pi * t / 15),
        -0.1 * sig.square(2 * np.pi * t / 15),
        # -0.2 * sig.square(2 * np.pi * t / 15),
        # -0.3 * sig.square(2 * np.pi * t / 15),
        0.1 * sig.square(2 * np.pi * t / 9),
        # 0.2 * sig.square(2 * np.pi * t / 9),
        # 0.3 * sig.square(2 * np.pi * t / 9),
        # -0.1 * sig.square(2 * np.pi * t / 9),
        -0.2 * sig.square(2 * np.pi * t / 9),
        # -0.3 * sig.square(2 * np.pi * t / 9),
        # 0.1 * sig.square(2 * np.pi * t / 7),
        # 0.2 * sig.square(2 * np.pi * t / 7),
        # 0.3 * sig.square(2 * np.pi * t / 7),
        # -0.1 * sig.square(2 * np.pi * t / 7),
        # -0.2 * sig.square(2 * np.pi * t / 7),
        # -0.3 * sig.square(2 * np.pi * t / 7),
    )

    # setpoint = np.full(t.shape, 0.30)  # Setpoint constant: "maintenir la bille a une position fixee"
    # setpoint = -0.30 * np.sin(2 * np.pi * t / 15)  # Setpoint = sinus
    setpoint = 0.25 * sig.square(2 * np.pi * t / 100)  # Setpoint = carre
    # setpoint = 0.25 * np.sin(2 * np.pi * t / 40)  # Setpoint = sinus


    # Decommenter les deux lignes ci-dessous pour lancer un fit du controleur PID sur la reference 'setpoint'
    # et pour le simulateur 'sim'
    # print(fit_pid(sim, setpoints_to_fit, init_values=np.array([51.25805776, -0.21727106, 7.67898543]),
    #               method="Powell"))
    # print(fit_pid(sim, setpoints_to_fit, init_values=np.array([10.31712585,  0.49838698,  3.88553031]),
    #               method="L-BFGS-B", bounds=((-20, 20), (-20, 20), (-20, 20))))
    # exit()

    # Valeurs de parametres PID obtenues par optimisation de l'erreur totale (lineaire, pas MSE)
    # sur un mix de setpoints (constant/sinus/carre).
    cont = Obj3PIDBBController(sim, 5.13051124e+01, -1.59963530e-02,  9.82885344e+00)

    cont.simulate(setpoint, n_steps=n_steps, init_state=np.array([0.2, -0.3]))

    fig, ((ax_pos), (ax_theta)) = plt.subplots(nrows=2, sharex=True)
    ax_pos.plot(t, setpoint, "ro--", linewidth=0.7, markersize=2, markevery=20, label="Setpoint [m]")
    ax_pos.plot(t, sim.all_y[:n_steps], "k-", label="Position [m]")
    ax_pos.plot(t, sim.all_y[:n_steps].flatten() - setpoint, "m--", linewidth=0.7, label="Error [m]")
    ax_pos.plot(t, np.full(t.shape, 0.775 / 2), color="grey", linestyle="--", linewidth=1, label="Bounds")
    ax_pos.plot(t, np.full(t.shape, -0.775 / 2), color="grey", linestyle="--", linewidth=1)
    ax_theta.plot(t, np.rad2deg(sim.all_u[:n_steps] - sim.params["theta_offset"]), color="navy",
                  linestyle="--", linewidth=0.7, label="Commanded angle [deg]")
    # ax_theta.plot(t, np.rad2deg(sim.all_u[:n_steps]), color="darkturquoise", linestyle="-",
    #               label="Actual offset angle [deg]")
    ax_theta.plot(t, np.full(t.shape, -50), color="grey", linestyle="--", linewidth=1)
    ax_theta.plot(t, np.full(t.shape, 50), color="grey", linestyle="--", linewidth=1)
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
