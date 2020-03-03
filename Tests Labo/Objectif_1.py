# -*- coding: utf-8 -*-

# Author: Eduardo Vannini
# Date: 01-03-2020

# Petit script pour ecrire des fichiers "de reference" pour l'objectif 1.
# Puisqu'on va utiliser la variable LabVIEW 'Ref' pour contenir l'angle de la poutre que l'on desire
# obtenir et puisque cette variable lit ses valeurs dans un fichier texte, il faut qu'on genere de tels
# fichiers. C'est ce qu'on fait ici.

if __name__ == "__main__":
    import numpy as np
    import os
    from Tests import Tests

    base_dir = "./Objectif 1/Tests Input"  # Repertoire contenant les fichiers d'input (angle alpha en degres)
    dt = 0.05                              # Periode d'echantillonnage [s]
    test_span = 30                         # Duree d'un test [s]

    # all_alphas = np.arange(-25, -20, 1)
    # for alpha in all_alphas:
    #     # On va recycler les fonctions d'ecriture de tests qu'on a utilisees pour la validation.
    #     # La fonction s'appelle 'write_theta' mais elle fonctionne tres bien pour ecrire des angles 'alpha' aussi.
    #     Tests.write_theta(os.path.join(base_dir, "obj_1_ref_{:.0f}.txt".format(alpha)),
    #                       np.full(int(test_span / dt), fill_value=alpha))

    alpha = np.linspace(-30, 30, int(test_span / dt))
    # On va recycler les fonctions d'ecriture de tests qu'on a utilisees pour la validation.
    # La fonction s'appelle 'write_theta' mais elle fonctionne tres bien pour ecrire des angles 'alpha' aussi.
    Tests.write_theta(os.path.join(base_dir, "obj_1_ref_-30_30.txt"), alpha)