# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 22-02-2020

import numpy as np
import itertools as it
from Simulator import Simulator
from BBSimulators import BBSimpleSimulator, BBAlphaSimulator, BBThetaSimulator


@Simulator.command_limiter(low_bound=np.deg2rad(-50), up_bound=np.deg2rad(50))
def my_command(timestep, params, all_t, all_u, all_y, dt):
    return 0


# dt=0.05s est pas terrible pour une simulation... Mais c'est ce qu'on a au labo.
sim = BBThetaSimulator(dt=0.001, buffer_size=1000)
print("[STATUS] Starting tests using 'dt' = {}s and 'buffer_size' = {}\n".format(sim.dt, sim.buffer_size))

# Sanity check #1:
#   - Setup: angle nul; balle au repos a la position p0 situee sur la poutre (L = 0.775; p0 dans [-L/2, L/2])
#   - Sortie attendue: la balle reste a la position p0
p0_to_test = (0, 0.2, -0.3)
for p0 in p0_to_test:
    print("[STATUS] Started sanity check #1 using p0 = {}m".format(p0))
    sim.simulate(command_func=lambda *args: 0, init_state=np.array([p0, 0]))
    if np.any(sim.all_x[:, 0] != p0) or np.any(sim.all_x[:, 1] != 0):
        print("[FAILURE] Sanity check #1 - 'all_x' failed the test for p0 = {}m".format(p0))
    if np.any(sim.all_y != p0):
        print("[FAILURE] Sanity check #1 - 'all_y' failed the test for p0 = {}m".format(p0))
    if np.any(sim.all_u != 0):
        print("[FAILURE] Sanity check #1 - 'all_u' failed the test for p0 = {}m".format(p0))
    print()

# Sanity check #2:
#   - Setup: angle nul, balle en position initiale p0 et avec une vitesse v0
#   - Sortie attendue: la balle ralentit jusqu'a s'arreter (frottement et/ou fin du tube)
# Note: Le pas de temps dt=0.05s est trop grand pour pouvoir faire des tests fiables dessus.
#       Ces tests-ci n'ont une valeur que si on a un dt suffisamment petit (ce qui permet un
#       meilleure stabilite des differences finies dans le modele)
p0_to_test = (0, 0.2, -0.3)
v0_to_test = (-0.1, 0, 0.3)
atol = 1e-02  # Tolerance pour les comparaisons d'ordre (<, >, etc.)
for p0, v0 in it.product(p0_to_test, v0_to_test):
    print("[STATUS] Started sanity check #2 using p0 = {}m and v0 = {}m/s (atol = {})".format(p0, v0, atol))
    sim.simulate(command_func=lambda *args: 0, init_state=np.array([p0, v0]))
    if np.any((sim.all_x[:, 0] - p0) * np.sign(v0) + atol < 0) or np.any(np.abs(np.diff(sim.all_x[:, 1])) - atol > 0):
        print("[FAILURE] Sanity check #2 - 'all_x' failed the test for p0 = {}m and v0 = {}m/s".format(p0, v0))
    if np.any((sim.all_y - p0) * np.sign(v0) + atol < 0):
        print("[FAILURE] Sanity check #2 - 'all_y' failed the test for p0 = {}m and v0 = {}m/s".format(p0, v0))
    if np.any(sim.all_u != 0):
        print("[FAILURE] Sanity check #2 - 'all_u' failed the test for p0 = {}m and v0 = {}m/s".format(p0, v0))
    print()


# Autres tests possibles (necessitent des donnees experimentales):
#   - Angle non-nul fixe, position initiale fixee, vitesse initiale nulle: comparaison du temps pour
#     atteindre le bout du tube
#   - Angle suivant un sinus pas trop "agite" (periode pas trop courte): comparaison des positions au cours du temps
#   - ...

# TODO:
# Ce que le modele ne fait pas encore:
#   - Validation de la transformation sur l'angle
#   - Fit par rapport aux donnees reelles (principalement: il faut trouver le coefficient de frottement)
#   - Fit par rapport aux bruits
#   - Il ne tient pas compte de l'inertie de la poutre (hypothese de controle instantane)
#       - Pour modeliser la "latence" de la commande: mettre un seuil sur sa derivee.

print("[STATUS] Done")
