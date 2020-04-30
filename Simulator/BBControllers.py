# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 29-02-2020

import numpy as np
import scipy.optimize as opt
import abc
import time
import matplotlib.pyplot as plt
import scipy.signal as sig

from Simulator import Simulator
from BBSimulators import BBThetaSimulator, BBObj7Simulator


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
        # Pas d'implementation car on est dans la classe abstraite. A implementer dans les sous-classes.
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
        # L'idee est simplement de "traduire" la 'control_law' en 'command_func' pour l'utiliser dans le simulateur.
        def command_func(timestep, params, all_t, all_u, all_y, dt):
            if timestep > 0:
                return self.control_law(setpoint[timestep], all_y[timestep - 1], dt, all_u[timestep - 1], self.flags)
            return self.control_law(setpoint[timestep], 0, dt, 0, self.flags)

        # On passe le tout a 'sim.simulate'.
        return self.sim.simulate(command_func, command_noise_func, output_noise_func, n_steps, init_state)


class PIDBBController(BBController):
    """
    Classe qui implemente une loi de controle de type PID pour le systeme Ball and Beam du projet P4 MAP.
    Aucun autre mecanisme n'est implemente ici: pas d'idiot-proofing, notamment.
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

        kp, ki, kd = self.kp, self.ki, self.kd          # A hardcoder dans LabVIEW
        theta_offset = self.sim.params["theta_offset"]  # A hardcoder dans LabVIEW
        err = pos - ref

        deriv_err = (err - self.flags[1]) / dt
        self.flags[0] += err * dt
        self.flags[1] = err
        integ_err = self.flags[0]

        # Il faut faire attention a bien tenir compte de l'offset
        return kp * err + ki * integ_err + kd * deriv_err - theta_offset


class FreeControlBBController(BBController):
    """
    Classe qui permet d'implementer un 'free-control'. L'argument 'ref' de la methode 'control_law' est ici
    utilise comme une commande de l'angle du moteur et pas comme un setpoint. Cette classe est utilisee pour
    l'interface permettant le test de notre idiot-proofing, dans le fichier 'Interface.py'.
    """

    def __init__(self, sim, using_idiot_proofing=True):
        super().__init__(sim)
        self.using_idiot_proofing = using_idiot_proofing

    @Simulator.command_limiter(low_bound=np.deg2rad(-50), up_bound=np.deg2rad(50))
    def control_law(self, ref, pos, dt, u_1, flags_1):
        # Ici, 'ref' est directement redirige vers la commande, apres un passage dans l'idiot-proofing si
        # ce mecanisme est active. 'ref' n'est donc PAS un setpoint, mais plutot un angle [rad] sur lequel
        # l'offset doit encore etre applique.
        # Memorisation de la position precedente avec le flag #3
        self.flags = flags_1
        theta_offset = self.sim.params["theta_offset"]  # A hardcoder dans LabVIEW
        raw_command = ref - theta_offset

        if not self.using_idiot_proofing:
            return raw_command

        # Idiot proofing
        speed = (pos - self.flags[3]) / dt  # Calcul de la vitesse a l'aide du flag #3
        self.flags[3] = pos  # Mise a jour du flag #3 pour l'iteration suivante
        pos_lim = 0.33       # Delimitation de la "Bypass zone" (situee entre pos_lim et le bord)
        buf_dist = 0.06      # Delimitation de la "Buffer zone" (buf_dist metres avant pos_lim)
        speed_lim = 0.025    # Limite de vitesse autorisee dans la "Bypass zone"

        # Gestion de la "Bypass zone": on fait sortir la bille de cette zone avec un angle plus ou moins grand
        # Principe: on veut faire atterir la bille dans la buffer zone, donc d'abord on l'arrete, puis on la
        # fait repartir avec une vitesse assez faible.
        if abs(pos) > pos_lim:
            # On adapte l'angle selon la necessite: on ne veut pas etre trop ferme, sinon ca va osciller
            if np.sign(speed) == np.sign(pos):
                angle = np.deg2rad(49)
            elif 0 <= abs(speed) <= speed_lim:
                angle = np.deg2rad(15) - (np.deg2rad(15) / speed_lim * abs(speed))
            else:
                angle = 0
            return np.sign(pos) * angle - theta_offset

        # Gestion de la "Buffer zone": cette zone sert a freiner la bille mais aussi comme zone buffer pour eviter que
        # le controleur et l'idiot-rpoofing se renvoient la bille sans cesse.
        # Note importante: Il n'y a une action de cette zone que quand la commande pousse la bille vers le bout du tube.
        if pos_lim - buf_dist <= abs(pos) <= pos_lim and np.sign(raw_command) == -np.sign(pos):
            if np.sign(speed) == np.sign(pos):
                # Freinage
                return np.sign(pos) * np.deg2rad(49) - theta_offset
            else:
                # Buffer
                return 0 - theta_offset

        # Si l'idiot proofing n'a pas effectue d'action particuliere, on retourne juste l'angle du controleur.
        return raw_command


class Obj5PIDBBController(BBController):
    """
    Classe qui implemente une loi de controle de type PID pour le systeme Ball and Beam du projet P4 MAP.
    Ce controleur implemente aussi une version du "idiot proofing".
    """

    def __init__(self, sim, kp, ki, kd, using_idiot_proofing=True):
        super().__init__(sim)
        self.using_idiot_proofing = using_idiot_proofing
        self.kp, self.ki, self.kd = kp, ki, kd

    @Simulator.command_limiter(low_bound=np.deg2rad(-50), up_bound=np.deg2rad(50))
    def control_law(self, ref, pos, dt, u_1, flags_1):
        # Modifie self.flags et retourne u
        # Attention, dans LabVIEW ref est donne en cm, mais ici on le fait en m.
        # De meme, les angles sont geres en degres dans LabVIEW et en radians ici.
        #
        # Integrale de l'erreur avec le flag #0
        # Memorisation de l'erreur precedente avec le flag #2
        # Memorisation de la position precedente avec le flag #3
        self.flags = flags_1

        # Phase "controleur"
        # Parametres du PID:
        kp, ki, kd = self.kp, self.ki, self.kd          # A hardcoder dans LabVIEW
        theta_offset = self.sim.params["theta_offset"]  # A hardcoder dans LabVIEW

        # Controle PID
        err = pos - ref
        deriv_err = (err - self.flags[2]) / dt
        self.flags[0] += err * dt
        self.flags[2] = err
        integ_err = self.flags[0]

        raw_command = kp * err + ki * integ_err + kd * deriv_err - theta_offset
        if not self.using_idiot_proofing:
            return raw_command

        # Idiot proofing
        speed = (pos - self.flags[3]) / dt  # Calcul de la vitesse a l'aide du flag #3
        self.flags[3] = pos                 # Mise a jour du flag #3 pour l'iteration suivante
        pos_lim = 0.33                      # Delimitation de la "Bypass zone" (situee entre pos_lim et le bord)
        buf_dist = 0.06                     # Delimitation de la "Buffer zone" (buf_dist metres avant pos_lim)
        speed_lim = 0.025                   # Limite de vitesse autorisee dans la "Bypass zone"

        # Gestion de la "Bypass zone": on fait sortir la bille de cette zone avec un angle plus ou moins grand
        # Principe: on veut faire atterir la bille dans la buffer zone, donc d'abord on l'arrete, puis on la
        # fait repartir avec une vitesse assez faible.
        if abs(pos) > pos_lim:
            # On adapte l'angle selon la necessite: on ne veut pas etre trop ferme, sinon ca va osciller
            if np.sign(speed) == np.sign(pos):
                angle = np.deg2rad(49)
            elif 0 <= abs(speed) <= speed_lim:
                angle = np.deg2rad(15) - (np.deg2rad(15) / speed_lim * abs(speed))
            else:
                angle = 0
            return np.sign(pos) * angle - theta_offset

        # Gestion de la "Buffer zone": cette zone sert a freiner la bille mais aussi comme zone buffer pour eviter que
        # le controleur et l'idiot-rpoofing se renvoient la bille sans cesse.
        # Note importante: Il n'y a une action de cette zone que quand la commande pousse la bille vers le bout du tube.
        if pos_lim - buf_dist <= abs(pos) <= pos_lim and np.sign(raw_command) == -np.sign(pos):
            if np.sign(speed) == np.sign(pos):
                # Freinage
                return np.sign(pos) * np.deg2rad(49) - theta_offset
            else:
                # Buffer
                return 0 - theta_offset

        # Si l'idiot proofing n'a pas effectue d'action particuliere, on retourne juste l'angle du controleur.
        return raw_command


class Obj7Controller(BBController):
    """
    Classe qui implemente un controleur assez simple pour les objectifs 6 et 7, tout en maintenant l'idiot-proofing
    developpe dans le controleur PID de l'objectif 5. La manoeuvre a effectuer pour ces deux objectifs necessite une
    etape de "positionnement" a vitesse non-contrainte, suivie d'une etape de "trajectoire" a vitesse contrainte.
    Pour l'etape de positionnement, on reutilise le controleur 'Obj5PIDController' muni de l'idiot-proofing. Pour
    l'etape de "trajectoire", en revanche, on utilise une nouvelle loi de commande qui permet de maintenir une vitesse
    adequate, comprise dans les limites specifiees par 'v_min' et 'v_max'.
    """
    def __init__(self, sim, kp, ki, kd, x_1, x_2, v_min, v_max, using_idiot_proofing=True):
        super().__init__(sim)
        self.using_idiot_proofing = using_idiot_proofing

        # Calcul du gain sur base de donnees experimentales de "ce qui fonctionne bien comme gain"
        k_dict = {4: 5, 5: 7, 6: 10, 7: 12, 8: 15, 9: 15, 10: 18, 11: 18, 12: 18, 13: 20, 14: 20, 15: 22, 16: 25,
                  17: 30, 18: 30, 19: 35, 20: 35, 21: 40, 22: 50, 23: 50, 24: 50, 25: 50, 26: 50}
        v_mean = 0.5 * (v_min + v_max) * 100
        best = np.inf
        for key in k_dict.keys():
            diff = abs(key - v_mean)
            if diff < best:
                best = diff
                self.k = np.deg2rad(k_dict[key])

        self.x_1, self.x_2, self.v_min, self.v_max = x_1, x_2, v_min, v_max
        self.kp, self.ki, self.kd = kp, ki, kd

    @Simulator.command_limiter(low_bound=np.deg2rad(-50), up_bound=np.deg2rad(50))
    def control_law(self, ref, pos, dt, u_1, flags_1):
        # Modifie self.flags et retourne u
        # Attention, dans LabVIEW ref est donne en cm, mais ici on le fait en m.
        # De meme, les angles sont geres en degres dans LabVIEW et en radians ici.
        # Memorisation de la position precedente avec le flag #0
        # Memorisation de l'etape courante avec le flag #1 (0 = "placement", 1 = "faire la trajectoire x1->x2")
        # Integrale de l'erreur avec le flag #2
        # Memorisation de l'erreur precedente avec le flag #3
        self.flags = flags_1

        speed = (pos - self.flags[0]) / dt  # Calcul de la vitesse a l'aide du flag #0
        self.flags[0] = pos                 # Mise a jour du flag #0 pour l'iteration suivante

        # Phase 1: "controleur"
        # Parametres du controleur (a hardcoder dans LabVIEW)
        k = self.k
        theta_offset = self.sim.params["theta_offset"]
        x_1, x_2, v_min, v_max = self.x_1, self.x_2, self.v_min, self.v_max
        delta_v = v_max - v_min

        # Calcul d'une position initiale permettant d'accelerer suffisamment avant d'atteindre 'x_1'
        k_sp = 0.95  # Coefficient de "securite" par rapport aux v_min et v_max
        start_pos = x_1 - np.sign(x_2 - x_1) * (v_min + k_sp * delta_v) ** 2 / 0.7
        if start_pos > 0:
            start_pos += 0.01
        if start_pos < 0:
            start_pos -= 0.01

        # En l'absence d'idiot-proofing, on limite 'start_pos' au niveau des extremites du tube
        if not self.using_idiot_proofing:
            start_pos = np.sign(start_pos) * min(start_pos, 0.7 / 2)
        # Sinon, si 'start_pos' est au-dela de la "buffer zone" de l'idiot-proofing, on la bride hors de celle-ci
        elif abs(start_pos) > 0.33 - 0.06:
            start_pos = (0.33 - 0.06) * np.sign(start_pos)

        # Calcul de la commande
        raw_command = 0 - theta_offset

        # Si on est en mode "placement"
        if self.flags[1] == 0:
            # Si on doit encore se positionner au niveau de 'start_pos'
            if abs(pos) < abs(start_pos) - 0.01 or pos * start_pos < 0:
                # Dans cette situation, la contrainte de vitesse ne s'applique pas
                # On ne doit pas forcement se stabiliser a 'start_pos', on peut donc utiliser un controleur plus simple
                # Controle PID
                kp, ki, kd = self.kp, self.ki, self.kd 
                err = pos - start_pos * 1.10
                deriv_err = (err - self.flags[3]) / dt
                self.flags[2] += err * dt
                self.flags[3] = err
                integ_err = self.flags[2]
                raw_command = kp * err + ki * integ_err + kd * deriv_err - theta_offset
            # Sinon, on bascule en mode "faire la trajectoire x_1 -> x_2"
            else:
                self.flags[1] = 1

        # Si on est en mode "faire la trajectoire x_1 -> x_2"
        if self.flags[1] == 1:
            if np.sign(x_2 - x_1) * pos > 0 and abs(pos) > abs(x_2):
                # Mission achevee, on peut donner une commande nulle parce qu'on ne doit plus faire quoi que ce soit
                raw_command = 0 - theta_offset
            else:
                # Calcul de la commande: nulle quand |speed| = k_sp * v_max, egale a +-k quand |speed| = k_sp * v_min
                raw_command = k * np.sign(x_1 - x_2) * (v_min + delta_v / 2 * (1 + k_sp) - abs(speed)) /\
                              (k_sp * delta_v) - theta_offset

        if not self.using_idiot_proofing:
            return raw_command

        # Phase 2: "Idiot proofing"
        pos_lim = 0.33     # Delimitation de la "Bypass zone" (situee entre pos_lim et le bord)
        buf_dist = 0.06    # Delimitation de la "Buffer zone" (buf_dist metres avant pos_lim)
        speed_lim = 0.025  # Limite de vitesse autorisee dans la "Bypass zone"

        # Gestion de la "Bypass zone": on fait sortir la bille de cette zone avec un angle plus ou moins grand
        # Principe: on veut faire atterir la bille dans la buffer zone, donc d'abord on l'arrete, puis on la
        # fait repartir avec une vitesse assez faible.
        if abs(pos) > pos_lim:
            # On adapte l'angle selon la necessite: on ne veut pas etre trop ferme, sinon ca va osciller
            if np.sign(speed) == np.sign(pos):
                angle = np.deg2rad(49)
            elif 0 <= abs(speed) <= speed_lim:
                angle = np.deg2rad(15) - (np.deg2rad(15) / speed_lim * abs(speed))
            else:
                angle = 0
            return np.sign(pos) * angle - theta_offset

        # Gestion de la "Buffer zone": cette zone sert a freiner la bille mais aussi comme zone buffer pour eviter que
        # le controleur et l'idiot-rpoofing se renvoient la bille sans cesse.
        # Note importante: Il n'y a une action de cette zone que quand la commande pousse la bille vers le bout du tube.
        if pos_lim - buf_dist <= abs(pos) <= pos_lim and np.sign(raw_command) == -np.sign(pos):
            if np.sign(speed) == np.sign(pos):
                # Freinage
                return np.sign(pos) * np.deg2rad(49) - theta_offset
            else:
                # Buffer
                return 0 - theta_offset

        # Si l'idiot proofing n'a pas effectue d'action particuliere, on retourne juste l'angle du controleur.
        return raw_command


def fit_pid(sim, setpoint_list, init_values=None, method=None, bounds=None):
    """
    Fonction permettant de faire une minimisation de l'erreur pour un controleur PID et sur un
    simulateur donne. Le simulateur est de type 'BBSimulator'. La minimisation se fait pour
    une suite de 'setpoints' donnee et n'a donc pas un caractere "general".

    Note: la minimisation utilise un controleur PID *sans* idiot-proofing. Cela permet d'obtenir les valeurs kp, ki et
          kd dans un premier temps et d'ensuite concevoir un idiot-proofing adequat.

    :param sim           : Objet de type 'BBSimulator' qui gere la simulation du systeme.
    :param setpoint_list : Liste d'array de points de reference (setpoints) sur lesquels la minimisation s'appuye.
    :param init_values   : Valeurs initiales pour Kp, Ki et Kd. Prises au hasard si 'init_values' = None.
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
        return tot_err

    # Selection de valeurs initiales aleatoires. Les ordres de grandeur ont ete precises afin d'accelerer
    # l'optimisation (on demarre plus proche d'une solution).
    if init_values is None:
        init_values = np.array([100, 0.01, 10]) * np.random.random(3)

    minimizer_kwargs = {"method": method, "bounds": bounds, "options": {"disp": True},
                        "callback": lambda xk: print("Params: {}\nMean error per setpoint: {}\n"
                                                     "".format(xk, err_func(xk) / len(setpoint_list)))}

    # Utilisation du basinhopping: on effectue plusieurs minimisations en partant de conditions initiales differentes
    # afin de ne pas se limiter a un seul minimum local. Le meilleur resultat est celui retourne.
    return opt.basinhopping(err_func, init_values, niter=5, minimizer_kwargs=minimizer_kwargs, disp=True)


def plot_simulation(t, n_steps, sim, setpoint):
    """
    Fonction permettant de faire un graphique detaille avec le resultat d'une simulation avec tentative de controle.

    :param t        : Vecteur contenant les valeurs des temps ou la simulation a ete effectuee [s];
    :param n_steps  : Nombre d'elements du vecteur 't' que l'on desire afficher (inferieur ou egal a t.size);
    :param sim      : Simulateur ayant ete utilise pour la simulation;
    :param setpoint : Setpoint ayant ete utilise pour la simulation;
    :return         : None.
    """
    fig, ((ax_pos), (ax_theta)) = plt.subplots(nrows=2, sharex=True)
    ax_pos.plot(t, setpoint, "ro--", linewidth=0.7, markersize=2, markevery=20, label="Setpoint [m]")
    ax_pos.plot(t, sim.all_y[:n_steps], "k-", label="Position [m]")
    ax_pos.plot(t, sim.all_y[:n_steps].flatten() - setpoint, "m--", linewidth=0.7, label="Error [m]")
    ax_pos.plot(t, np.full(t.shape, 0.775 / 2), color="grey", linestyle="--", linewidth=1, label="Bounds")
    ax_pos.plot(t, np.full(t.shape, -0.775 / 2), color="grey", linestyle="--", linewidth=1)
    ax_theta.plot(t, np.rad2deg(sim.all_u[:n_steps] - sim.params["theta_offset"]), color="navy",
                  linestyle="--", linewidth=0.7, label="Commanded angle [deg]")
    ax_theta.plot(t, np.full(t.shape, -50), color="grey", linestyle="--", linewidth=1)
    ax_theta.plot(t, np.full(t.shape, 50), color="grey", linestyle="--", linewidth=1)
    fig.legend(loc="right")
    ax_pos.grid()
    ax_theta.grid()
    ax_pos.set_xlabel("Time [s]")
    ax_pos.set_ylabel("Position [m]")
    ax_theta.set_xlabel("Time [s]")
    ax_theta.set_ylabel("Angle [deg]")


def launch_fit_pid(t, sim):
    """
    Fonction utilitaire permettant de lancer le fit du controleur PID (sans idiot-proofing) avec des parametres
    predefinis. Cette fonction ne retourne rien, les resultats du fit doivent etre lus sur la console.

    :param t   : Vecteur contenant les pas de temps ou la simulation devra etre effectuee [s];
    :param sim : Simulateur a utiliser lors du calcul de l'erreur;
    :return    : None.
    """
    # Liste des setpoints sur lesquels on se base pour fit le PID.
    # Le fit s'effectue sur un controleur depourvu d'idiot-proofing.
    setpoints_to_fit = (
        # Quelques trajectoires constantes:
        np.zeros(t.shape),
        np.full(t.shape, -0.30),
        np.full(t.shape, -0.10),
        np.full(t.shape, 0.20),
        # Quelques trajectoires sinusoidales:
        0.1 * np.sin(2 * np.pi * t / 15),
        0.3 * np.sin(2 * np.pi * t / 15),
        -0.2 * np.sin(2 * np.pi * t / 15),
        -0.1 * np.sin(2 * np.pi * t / 9),
        # Quelques trajectoires carrees:
        0.1 * sig.square(2 * np.pi * t / 15),
        0.3 * sig.square(2 * np.pi * t / 15),
        -0.2 * sig.square(2 * np.pi * t / 15),
        0.1 * sig.square(2 * np.pi * t / 9),
    )

    start_time = time.time()
    print(fit_pid(sim, setpoints_to_fit, init_values=None, method="BFGS"))
    exec_time = time.time() - start_time
    print("Fit completed in {} s".format(exec_time))


# La suite du code ne sera executee que si le fichier 'BBControllers.py' est lance directement. Elle ne le sera pas si
# ce fichier est utilise comme import dans un autre fichier. La section ci-dessous sert de code de demonstration pour
# l'utilisation des classes implementees ci-dessus.
if __name__ == "__main__":
    # Definition du vecteur de temps: la simulation durera de t = 0s a t = 30s, et on utilise un pas de temps dt = 50ms
    t = np.arange(0, 10, 0.05)

    # Dans le cas ou l'on teste ou l'objectif 7, decommenter cette fonction et definir la perturbation
    # def perturbation(a_desired, x, v, alpha, t):
    #     """
    #     Fonction de perturbation qui permet de forcer l'acceleration de la bille. L'acceleration non-forcee est
    #     'a_desired'. Pour rendre cette fonction sans effet, faire 'return a_desired'.
    #
    #     :param a_desired : Valeur de l'acceleration de la bille issue du modele physique [m/s^2]
    #     :param x         : Position de la bille [m]
    #     :param v         : Vitesse de la bille [m/s]
    #     :param alpha     : Angle du servo (pas de la poutre!) [rad]
    #     :param t         : Temps depuis le debut de la simulation [s]
    #     :return          : Valeur de l'acceleration forcee de la bille [m/s^2]
    #     """
    #     # Decommenter pour tester diverses perturbations
    #
    #     # Pas de perturbation
    #     return a_desired
    #
    #     # Perturbation d'une duree d'une seconde, au debut de la simulation
    #     # if 0 <= t < 1:
    #     #     return a_desired * 1.10
    #     # else:
    #     #     return a_desired

    # Choix du simulateur (choisir 'BBThetaSimulator' pour tous les objectifs sauf le 7)
    sim = BBThetaSimulator(dt=0.05, buffer_size=t.size + 1)
    # sim = BBObj7Simulator(perturbation, dt=0.05, buffer_size=t.size + 1)

    # Decommenter les deux lignes ci-dessous pour lancer un fit du controleur PID. Ceci est fait pour etre utilise
    # avec le simulateur 'BBThetaSimulator'. Ne pas utiliser avec 'BBObj7Simulator'.
    # launch_fit_pid(t, sim)
    # exit()

    # Pas besoin de s'occuper de ceci, c'est un raccourci
    n_steps = t.size

    # Il faut choisir un setpoint. Dans le cas des objectifs 3, 6 et 7, le setpoint importe peu et il ne sert qu'a
    # determiner la duree de la simulation. Dans le cas des objectifs 4 et 5, le setpoint est la trajectoire que l'on
    # va tenter de suivre. Decommenter pour choisir:
    setpoint = np.full(t.shape, 0.3)                 # Setpoint constant
    # setpoint = 0.15 * np.sin(2 * np.pi * t / 12)      # Setpoint = sinus
    # setpoint = 0.15 * sig.square(2 * np.pi * t / 20)  # Setpoint = carre
    # setpoint = 0.5 * np.sin(2 * np.pi * t / 40)       # Setpoint = sinus
    # setpoint = 0.25 * np.sin(2 * np.pi * t * np.power(np.exp(t), 0.005) / 20)  # Sinus de periode decroissante
    # setpoint = 0.25 * np.sin(2 * np.pi * t * np.power(np.exp(t), 0.01) / 20)  # Sinus de periode decroissante

    # Pour tester l'objectif 3, 4 ou 5, decommenter les lignes ci-dessous:
    # kp, ki, kd = 1.88700101e+01, -1.93159758e-03, 7.47877146e+00
    # cont = Obj5PIDBBController(sim, kp, ki, kd, using_idiot_proofing=True)

    # Test du controleur des objectifs 6 et 7:
    # Pour v_min = 0, ne pas descendre sous v_max = 0.03, car il s'agit de vitesses extremement lentes qui font
    # que la bille s'arrete constamment avec le frottement statique. Ca "fonctionne" mais il y a beaucoup d'oscillations
    # dues a la difficulte de la tache.
    # De meme, la vitesse maximale que la bille peut atteindre est 8 cm/s dans ce simulateur. Imposer v_min > 0.08
    # bloque donc le systeme.
    # Pour tester l'objectif 6 ou l'objectif 7, decommenter les lignes ci-dessous:
    x_1, x_2, v_min, v_max = 0.0, 0.32, 0.21, 0.25
    kp, ki, kd = 1.88700101e+01, -1.93159758e-03, 7.47877146e+00
    cont = Obj7Controller(sim, kp, ki, kd, x_1, x_2, v_min, v_max, using_idiot_proofing=True)

    # Lancement de la simulation avec controle. L'etat initial [position, vitesse] (unites SI) peut etre modifie:
    cont.simulate(setpoint, n_steps=n_steps, init_state=np.array([-0, 0]))

    # Affichage d'un graphe decrivant le resultat de la simulation
    plot_simulation(t, n_steps, sim, setpoint)
    plt.show()
