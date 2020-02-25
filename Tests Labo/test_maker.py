# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import csv
import io


def write_theta(out_path, theta):
    np.savetxt(out_path, theta, fmt="%.7e")
    with open(out_path, "r") as f:
        old = f.read()
    with open(out_path, "w") as f:
        f.write(old.replace(".", ","))

def plot_test(in_path, dt, t_max):
    with open(in_path, "r") as f:
        theta = np.array(list(map(lambda s: float(s.replace(",", ".")), f.read().split())))
    plt.plot(np.arange(0, t_max, dt), theta)

# Test #1
#   - Angle constant
#   - Position initiale qcq
def write_test_1(out_path, angle_deg, dt, t_max):
    t = np.arange(0, t_max, dt)
    theta = np.full(t.shape, fill_value=angle_deg)
    write_theta(out_path, theta)

# Test #2
#   - Angle evolue lineairement d'une valeur a une autre
#   - Position initiale qcq, vitesse initiale nulle
def write_test_2(out_path, angle_deg_1, angle_deg_2, dt, t_max):
    theta = np.arange(angle_deg_1, angle_deg_2, dt)
    write_theta(out_path, theta)

# Test #3
#   - Angle = sinus d'amplitude 'a' (degres) et de periode 'p' (seconds) en fonction de t
#   - Position initiale qcq, vitesse initiale nulle
def write_test_3(out_path, a, p, dt, t_max):
    t = np.arange(0, t_max, dt)
    theta = a * np.sin(2 * np.pi / p * t)
    write_theta(out_path, theta)

# Test #4:
#   On met la bille au bout du tube avec un certain angle 'a1' [deg] pendant t2 [s] puis on met un angle
#   assez grand 'a2' [deg] pendant t1 [s] pour donner de la vitesse puis on met l'angle a zero et on regarde
#   en combien de temps la bille s'arrete.
def write_test_4(out_path, a1, a2, t1, t2, dt, t_max):
    t = np.arange(0, t_max, dt)
    theta = np.zeros(t.shape)
    theta[:int(t1 / dt)] = a1
    theta[int(t1 / dt):int(t1 / dt) + int(t2 / dt)] = a2
    theta[int(t1 / dt) + int(t2 / dt):] = 0
    write_theta(out_path, theta)


if __name__ == "__main__":
    # Periode d'echantillonnage: 0.05 s
    DT = 0.05

    # Duree des tests: 60s
    T_MAX = 60

    # Angles limites: -50deg et +50deg (pour le servo)

    # for angle_deg in range(-40, 40, 20):
    #     write_test_1("./test_1_{}.txt".format(angle_deg), angle_deg, DT, T_MAX)
    #
    # for angle_deg_1, angle_deg_2 in it.product([-40, 0, 40], [-20, 0, 20]):
    #     write_test_2("./test_2_{}_{}.txt".format(angle_deg_1, angle_deg_2), angle_deg_1, angle_deg_2, DT, T_MAX)
    #
    # for a, p in it.product([10, 20, 30, 40], [5, 10, 20]):
    #     write_test_3("./test_3_{}_{}.txt".format(a, p), a, p, DT, T_MAX)

    # for a1, a2, t1, t2 in [[-30, 30, 20, 5], [-30, 30, 20, 10], [-30, 30, 20, 20],
    #                        [-40, 40, 20, 5], [-40, 40, 20, 10], [-40, 40, 20, 20]]:
    #     write_test_4("./test_4_{}_{}_{}_{}.txt".format(a1, a2, t1, t2), a1, a2, t1, t2, DT, T_MAX)

    # plot_test("./test_4_-40_40_20_10.txt", DT, T_MAX)
    # plt.show()
