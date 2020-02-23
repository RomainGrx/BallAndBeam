# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict

import pymoskito as pm


class MyPIDController(pm.Controller):
    """
    PID Controller
    """
    public_settings = OrderedDict([("Kp", -1),
                                   ("Ki", 0),
                                   ("Kd", 0),
                                   ("tick divider", 1)])
    last_time = 0

    def __init__(self, settings):
        # add specific "private" settings
        settings.update(input_order=0)
        settings.update(input_type="system_output")
        pm.Controller.__init__(self, settings)

        # define variables for data saving in the right dimension
        self.e_old = 0
        self.integral_old = 0
        self.last_u = 0
        self.output = 0

    def _control(self, time, trajectory_values=None, feedforward_values=None, input_values=None, **kwargs):
        # input abbreviations
        x = input_values
        yd = trajectory_values

        # step size
        dt = time - self.last_time
        # save current time
        self.last_time = time

        if dt != 0:
            # error
            e = x - yd
            integral = e * dt + self.integral_old
            differential = (e - self.e_old) / dt

            self.output = (self._settings["Kp"] * e
                           + self._settings["Ki"] * integral
                           + self._settings["Kd"] * differential)

            # save data for new calculation
            self.e_old = e
            self.integral_old = integral
            u = self.output
        else:
            u = self.last_u
        self.last_u = u
        return u
