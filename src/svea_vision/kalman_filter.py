# Not a ROS-node, only Kalman Filter
from filterpy.kalman import KalmanFilter
import numpy as np
from math import sin, cos

class KF(KalmanFilter):
    def __init__(self, id, init_pos: list, init_v: float, init_phi: float, frequency_of_measurements: float = 14.5):
        """ Kalman Filter implementation that also use the 
            id to keep track of separate measurements. 
             - The state is: [x,y,v,phi]"""
        super().__init__(dim_x=4, dim_z=4)

        # Set class attributes
        self.id = id
        self.process_variance = 0.13
        self.covariance = 2
        self.measurement_variance = 1  # Actual camera measurement noise (TBD)
        self.v_measurement_var = 0.003  # Assumed small
        self.phi_measurement_var = 0.003  # Assumed small
        self.dt = 1/frequency_of_measurements

        # Specify/initialize the Kalman parameters
        self.x = np.array([*init_pos, init_v, init_phi])

        # Control model (based on previous locations)
        self.F = np.array([[1, 0, self.dt*cos(self.x[3]), 0],
                           [0, 1, self.dt*sin(self.x[3]), 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Process uncertainty/noise
        self.Q = self.process_variance * np.eye(len(self.x), len(self.x))

        # Covariance matrix
        self.P = self.covariance*np.eye(len(self.x), len(self.x))

        # Measurement noise of the camera
        self.R = np.array([[self.measurement_variance, 0, 0, 0],
                           [0, self.measurement_variance, 0, 0],
                           [0, 0, self.v_measurement_var, 0],
                           [0, 0, 0, self.phi_measurement_var]])

        # Measurement model
        self.H = np.eye(len(self.x), len(self.x))
