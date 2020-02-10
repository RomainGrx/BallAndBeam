# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 09-02-2020

import numpy as np
import scipy.signal as sig
import control as ct
import control.matlab as matlab
import matplotlib.pyplot as plt


# Modelisation du systeme de la slide 42/48 de l'intro du projet en ajoutant une composante de controle
# en boucle fermee (suite de 'example_open_loop.py')

# Open-loop: 5 * (dy^2/dt^2)(t) + 4 * (dy/dt)(t) + 3 * y(t) = 2 * u(t)
# Avec y(t) la sortie au temps t et u(t) la commande au temps t

# On modelise ce systeme en temps discret (dt = 50ms, comme pour le ball and beam)
dt = 0.05

# On utilise la fonction de transfert (facile a calculer), puis on transforme en representation
# d'etat, car c'est necessaire pour ct.iosys.LinearIOSystem.
# Note: etant donne qu'on travaille en temps discret (dt), la fonction de transfert provient de la
# transformee en z du systeme discretise. Mais ca ne change rien en pratique car les coefficients
# sont les memes que ceux de la transformee de Laplace du systeme non discretise.
syst_open_tf = ct.tf(np.array([2]), np.array([5, 4, 3]), dt)
syst_open_ss = ct.tf2ss(syst_open_tf)

# Closed-loop: proof of concept avec un controleur PID discretise.
# Coefficients du controleur PID, a determiner par essai-erreur.
# Note:
#   - Un systeme trop "nerveux" <=> kp est trop grand (essayer en changeant kp!)
#   - Trop de delai pour atteindre le setpoint <=> ki est trop petit
#   - Effet de bord lors d'un changement de setpoint <=> ki est trop grand
kp = pow(10, -1)
ki = pow(10, 1.5)
kd = -pow(10, -2.5)

# Discretisation du controleur PID selon: http://portal.ku.edu.tr/~cbasdogan/Courses/Robotics/projects/Discrete_PID.pdf
control_prop_tf = ct.tf(np.array([kp]), np.array([1]), dt)
control_inte_tf = ct.tf(np.array([ki * dt, ki * dt]), np.array([2, -2]), dt)
control_deri_tf = ct.tf(np.array([kd, -kd]), np.array([dt, 0]), dt)

# Le controleur PID est la somme des trois sous-controleurs: P, I et D.
control_pid_tf = ct.parallel(control_prop_tf, control_inte_tf, control_deri_tf)

# A nouveau, on a besoin d'une representation d'etat pour utiliser ct.iosys.LinearIOSystem.
control_pid_ss = ct.tf2ss(control_pid_tf)

# Dans ce cas simplifie, on admet que le feedback est exact
# (On pourrait omettre ceci dans ce cas precis, mais autant le mettre pour savoir que ca existe).
feedback_ss = ct.tf2ss(np.array([1]), np.array([1]), dt)

# On rassemble tout (open-loop + controleur + feedback)
# sign=-1 indique une retroaction negative
model = ct.feedback(ct.series(control_pid_ss, syst_open_ss), feedback_ss, sign=-1)

# On transforme le modele en sa version "I/O" pour pouvoir lui donner des entrees et recuperer des sorties.
model_io = ct.iosys.LinearIOSystem(model)

# Analyse de la reponse du systeme a un signal (sinus, carre, etc) en entree pendant 10s.
t_in = np.arange(0, 10, dt)

# y_sp est le "setpoint" = la valeur vers laquelle on veut amener la sortie y(t).
# Decommenter pour tester diverses fonctions de setpoint
# y_sp = np.full(t_in.shape, 0.42)
# y_sp = np.sin(t_in)
y_sp = sig.square(t_in, duty=0.5)

# On fourni le setpoint voulu en entree du systeme *global*
# Attention, ceci est la commande du systeme en boucle fermee et pas du systeme en en boucle ouverte!
# (celle du systeme en boucle ouverte est geree par le controleur et pas par nous)
u_in = y_sp

# On recupere le signal y(t) mesure et on affiche le tout.
t_out, y = ct.input_output_response(model_io, T=t_in, U=u_in, squeeze=True)

plt.plot(t_in, u_in, "k-", lw=2, label="Setpoint")
plt.plot(t_out, y, "r-", lw=1, label="Output")
plt.plot(t_out, y_sp - y, "--", lw=1, label="Error")
plt.title("Behaviour of PID-controlled closed-loop system")
plt.grid()
plt.xlabel("Time t[s]")
plt.ylabel("e.g. Voltage u[V]")
plt.legend()
plt.show()
