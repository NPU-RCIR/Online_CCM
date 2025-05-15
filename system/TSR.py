'''
@Filename       : TSR.py
@Description    : 
@Create_Time    : 2025/02/17 11:03:35
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import torch
import numpy as np
import casadi as ca
from tqdm import tqdm
from system.Base import Base


class TSR(Base):
    def __init__(self):
        super().__init__()
        self.state_dim = 4
        self.u_dim = 2
        self.delta_t = 0.01

    def dynamics(self,t,x,u):
        """nominal dynamics of tether system

        Args:
            t (float): time
            x (array): state
            u (array): control input

        Returns:
            array: derivative of state(dot x)
        """
        if isinstance(x, np.ndarray):
            dx = np.zeros(self.state_dim)

            theta,length,d_theta,d_length = x[0],x[1],x[2],x[3]

            dx[0] = d_theta
            dx[1] = d_length
            dx[2] = -2*(d_length/(length+1))*(d_theta+1)-1.5*np.sin(2*theta) + u[0]
            dx[3] = (length+1)*(d_theta+1)**2-(length+1)*(1-3*(np.cos(theta))**2) - (u[1] + 3)
        if isinstance(x, torch.Tensor):
            dx = torch.zeros(self.state_dim)

            theta,length,d_theta,d_length = x[0],x[1],x[2],x[3]

            dx[0] = d_theta
            dx[1] = d_length
            dx[2] = -2*(d_length/(length+1))*(d_theta+1)-1.5*torch.sin(2*theta) + u[0]
            dx[3] = (length+1)*(d_theta+1)**2-(length+1)*(1-3*(torch.cos(theta))**2) - (u[1] + 3)

        return dx
    
    def dynamics_disturbance(self,t,x,u):
        """dynamics of tether system subjected to disturbance

        Args:
            t (float): time
            x (array): state
            u (array): control input

        Returns:
            array: derivative of state(dot x)
        """
        dx = np.zeros(self.state_dim)

        theta,length,d_theta,d_length = x[0],x[1],x[2],x[3]

        dx[0] = d_theta
        dx[1] = d_length
        dx[2] = -2*(d_length/(length+1))*(d_theta+1)-1.5*np.sin(2*theta) + u[0] + 0.3*(np.cos(t)+np.sin(x[2]))
        dx[3] = (length+1)*(d_theta+1)**2-(length+1)*(1-3*(np.cos(theta))**2) - (u[1] + 3 + 0.2*(np.cos(x[1])))

        return dx

    
    def dynamics_casadi(self,t,x,u):
        """tether dynamics(contious) using casadi SX symbol

        Args:
            t (float): time
            x (array): current state,[theta,dtheta,l,dl]
            u (array): current control

        Returns:
            array: next step state
        """
        dx = ca.SX.sym("dx", self.state_dim)

        theta,length,d_theta,d_length = x[0],x[1],x[2],x[3]

        dx[0] = d_theta
        dx[1] = d_length
        dx[2] = -2*(d_length/(length+1))*(d_theta+1)-1.5*np.sin(2*theta) + u[0]
        dx[3] = (length+1)*(d_theta+1)**2-(length+1)*(1-3*(np.cos(theta))**2) - (u[1] + 3)

        return dx
    
    def model4CCM_f(self,x):
        """ f function for CCM

        Args:
            x (bs x state_dim x 1): state

        Returns:
            f (bs x state_dim x 1): f function for CCM
        """

        bs = x.shape[0]

        f = torch.zeros(bs, self.state_dim,1).type(x.type())

        theta,length,d_theta,d_length = x[:,0,0],x[:,1,0],x[:,2,0],x[:,3,0]

        f[:,0,0] = d_theta
        f[:,1,0] = d_length
        f[:,2,0] = -2*(d_length/(length+1))*(d_theta+1) - 3*torch.sin(theta)*torch.cos(theta)
        f[:,3,0] = (length+1)*((d_theta+1)**2+3*(torch.cos(theta))**2-1) - 3 

        return f
    
    def model4CCM_B(self,x):
        """B function for CCM

        Args:
            x (bs x state_dim x 1): state

        Returns:
            B (bs x state_dim x u_dim): B function for CCM
        """
        bs = x.shape[0]

        B = torch.zeros(bs,self.state_dim,self.u_dim).type(x.type())

        B[:,2,0] = 1
        B[:,3,1] = -1

        return B