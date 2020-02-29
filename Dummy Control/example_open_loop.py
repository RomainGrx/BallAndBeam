# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 09-02-2020

import numpy as np
import control as ct
import matplotlib.pyplot as plt


# Modelisation du systeme de la slide 42/48 en boucle ouverte (sans controle)
# Open-loop: 5 * (dy^2/dt^2)(t) + 4 * (dy/dt)(t) + 3 * y(t) = 2 * u(t)
# Avec y(t) la sortie au temps t et u(t) la commande au temps t

# On modelise ce systeme en temps discret (dt = 50ms, comme pour le ball and beam)
dt = 0.05

# On utilise la fonction de transfert (facile a calculer), puis on transforme en representation
# d'etat, car c'est necessaire pour ct.iosys.LinearIOSystem.
syst_open_tf = ct.tf(np.array([2]), np.array([5, 4, 3]), dt)
syst_open_ss = ct.tf2ss(syst_open_tf)

# Creation d'un objet input/output pour modeliser la boucle ouverte.
syst_open_io = ct.iosys.LinearIOSystem(syst_open_ss)

# Analyse de la reponse du systeme a un signal en entree pendant 10s.
t_in = np.arange(0, 10, dt)
u_in = np.sin(t_in)
t_out, y_open = ct.input_output_response(syst_open_io, T=t_in, U=u_in, squeeze=True)

plt.plot(t_in, u_in, label="Command u(t)")
plt.plot(t_out, y_open, label="Output y(t)")
plt.title("Response of the open-loop system to a sinewave")
plt.grid()
plt.xlabel("Time t[s]")
plt.ylabel("e.g. Voltage u[V]")
plt.legend()
plt.show()
