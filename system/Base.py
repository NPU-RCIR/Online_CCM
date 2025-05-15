'''
@Filename       : base.py
@Description    : 
@Create_Time    : 2025/02/17 15:28:03
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import torch
import numpy as np
import casadi as ca
from tqdm import tqdm

class Base():
    def __init__(self):
        self.state_dim = 1
        self.u_dim = 1
        self.delta_t = 0.01
    
    def dynamics(self,t,x,u):
        """dynamics of tether system

        Args:
            t (float): time
            x (array): state
            u (array): control input

        Returns:
            array: derivative of state(dot x)
        """

        raise NotImplementedError
    
    def dynamics_disturbance(self,t,x,u):

        pass

    def discrete_dynamics(self,t,x,u):
        """using runge-kutta4 method for discreting dynamics 

        Args:
            t (float): time
            x (array): state
            u (array): control input

        Returns:
            array: next state
        """
        k1 = self.dynamics(t,x,u)
        k2 = self.dynamics(t,x + k1*self.delta_t/2,u)
        k3 = self.dynamics(t,x + k2*self.delta_t/2,u)
        k4 = self.dynamics(t,x + k1*self.delta_t,u)
        return (x + (self.delta_t/6)*(k1+2*k2+2*k3+k4))
    
    def discrete_dynamics_disturbance(self,t,x,u):
        """using runge-kutta4 method for discreting dynamics 

        Args:
            t (float): time
            x (array): state
            u (array): control input

        Returns:
            array: next state
        """
        k1 = self.dynamics_disturbance(t,x,u)
        k2 = self.dynamics_disturbance(t,x + k1*self.delta_t/2,u)
        k3 = self.dynamics_disturbance(t,x + k2*self.delta_t/2,u)
        k4 = self.dynamics_disturbance(t,x + k1*self.delta_t,u)
        return (x + (self.delta_t/6)*(k1+2*k2+2*k3+k4))
    
    def dynamics_casadi(self, t, x, u):
        """tether dynamics(contious) using casadi SX symbol

        Args:
            t (float): time
            x (array): current state,[theta,dtheta,l,dl]
            u (array): current control

        Returns:
            array: next step state
        """

        pass

    def discrete_dynamics_casadi(self, t, x, u):
        """using runge-kutta4 method for discreting tether dynamics using casadi SX symbol
        Args:
            t (float): time
            x (array): current state,[theta,dtheta,l,dl]
            u (array): current control

        Returns:
            array: next step state
        """
        k1 = self.dynamics_casadi(t, x, u)
        k2 = self.dynamics_casadi(t, x + k1 * self.delta_t / 2, u)
        k3 = self.dynamics_casadi(t, x + k2 * self.delta_t / 2, u)
        k4 = self.dynamics_casadi(t, x + k1 * self.delta_t, u)
        return x + (self.delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def data_collection(self, n_step, n_traj):
        """collecting training data

        Args:
            n_step (int): steps of each trajectory
            n_traj (int): number of trajectory

        Returns:p
            array: trajectory
        """
        pass

    def model4CCM_f(self, x):
        """f function for CCM

        Args:
            x (tensor): state

        Returns:
            tensor: derivative of state
        """
        pass

    def model4CCM_B(self, x):
        """B function for CCM

        Args:
            x (tensor): state

        Returns:
            tensor: control matrix
        """
        pass
