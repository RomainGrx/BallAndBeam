# -*- coding: utf-8 -*-
import pymoskito as pm

# import custom modules
import model
import controller


if __name__ == '__main__':
    # register model
    pm.register_simulation_module(pm.Model, model.BallBeamModel)

    # register controller
    pm.register_simulation_module(pm.Controller, controller.MyPIDController)

    # start the program
    pm.run()
