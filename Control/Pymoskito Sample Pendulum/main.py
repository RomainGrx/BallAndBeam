# -*- coding: utf-8 -*-
import pymoskito as pm

# import custom modules
import model


if __name__ == '__main__':
    # register model
    pm.register_simulation_module(pm.Model, model.PendulumModel)

    # start the program
    pm.run()