'''
@Filename       : PVTOL.py
@Description    : 
@Create_Time    : 2025/02/17 11:34:49
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import torch
import numpy as np
import casadi as ca
from tqdm import tqdm
from system.Base import Base


class PVTOL(Base):
    def __init__(self):
        super().__init__()
        self.state_dim = 6
        self.u_dim = 2
        self.delta_t = 0.01

        self.mass = 0.486
        self.J = 0.00383
        self.g = 9.81
        self.l = 0.25
    
    def dynamics(self,t,x,u):
        """dynamics of PVTOl system

        Args:
            t (float): time
            x (array): state
            u (array): control input

        Returns:
            array: derivative of state(dot x)
        """
        if isinstance(x, np.ndarray):
            dx = np.zeros(self.state_dim)

            px,pz,phi,vx,vz,dot_phi = x[0],x[1],x[2],x[3],x[4],x[5]

            dx[0] = vx*np.cos(phi) - vz*np.sin(phi)
            dx[1] = vx*np.sin(phi) + vz*np.cos(phi)
            dx[2] = dot_phi
            dx[3] = vz*dot_phi - self.g*np.sin(phi)
            dx[4] = -vx*dot_phi - self.g*np.cos(phi) + 1/self.mass*(u[0] + u[1])
            dx[5] = self.l/self.J*(u[0] - u[1])
        if isinstance(x, torch.Tensor):
            dx = torch.zeros(self.state_dim)

            px,pz,phi,vx,vz,dot_phi = x[0],x[1],x[2],x[3],x[4],x[5]

            dx[0] = vx*torch.cos(phi) - vz*torch.sin(phi)
            dx[1] = vx*torch.sin(phi) + vz*torch.cos(phi)
            dx[2] = dot_phi
            dx[3] = vz*dot_phi - self.g*torch.sin(phi)
            dx[4] = -vx*dot_phi - self.g*torch.cos(phi) + 1/self.mass*(u[0] + u[1])
            dx[5] = self.l/self.J*(u[0] - u[1])

        return dx
    
    def dynamics_disturbance(self, t, x, u):
        """dynamics of PVTOL system subjected to disturbance

        Args:
            t (float): time
            x (array): state
            u (array): control input

        Returns:
            array: derivative of state(dot x)
        """

        dx = np.zeros(self.state_dim)

        px,pz,phi,vx,vz,dot_phi = x[0],x[1],x[2],x[3],x[4],x[5]

        dx[0] = vx*np.cos(phi) - vz*np.sin(phi)
        dx[1] = vx*np.sin(phi) + vz*np.cos(phi)
        dx[2] = dot_phi
        dx[3] = vz*dot_phi - self.g*np.sin(phi)
        dx[4] = -vx*dot_phi - self.g*np.cos(phi) + 1/self.mass*(u[0] + u[1]) + 4*(np.cos(t)+np.sin(x[1])+np.cos(x[3]))
        dx[5] = self.l/self.J*(u[0] - u[1]) + 2*(np.sin(x[0])+np.cos(x[4]))

        return dx
    
    def dynamics_casadi(self,t,x,u):
        """PVTOL dynamics(contious) using casadi SX symbol

        Args:
            t (float): time
            x (array): current state,[theta,dtheta,l,dl]
            u (array): current control

        Returns:
            array: next step state
        """
        dx = ca.SX.sym("dx", self.state_dim)

        px,pz,phi,vx,vz,dot_phi = x[0],x[1],x[2],x[3],x[4],x[5]

        dx[0] = vx*ca.cos(phi) - vz*ca.sin(phi)
        dx[1] = vx*ca.sin(phi) + vz*ca.cos(phi)
        dx[2] = dot_phi
        dx[3] = vz*dot_phi - self.g*ca.sin(phi)
        dx[4] = -vx*dot_phi - self.g*ca.cos(phi) + 1/self.mass*(u[0] + u[1])
        dx[5] = self.l/self.J*(u[0] - u[1])

        return dx
    
    def model4CCM_f(self,x):
        """ f function for CCM

        Args:
            x (bs x state_dim x 1): state

        Returns:
            f (bs x state_dim x 1): f function for CCM
        """

        bs = x.shape[0]

        p_x, p_z, phi, v_x, v_z, dot_phi = [x[:,i,0] for i in range(self.state_dim)]
        
        f = torch.zeros(bs, self.state_dim, 1).type(x.type())

        f[:, 0, 0] = v_x * torch.cos(phi) - v_z * torch.sin(phi)
        f[:, 1, 0] = v_x * torch.sin(phi) + v_z * torch.cos(phi)
        f[:, 2, 0] = dot_phi
        f[:, 3, 0] = v_z * dot_phi - self.g * torch.sin(phi)
        f[:, 4, 0] = - v_x * dot_phi - self.g * torch.cos(phi)
        f[:, 5, 0] = 0

        return f
    
    def model4CCM_B(self,x):
        """B function for CCM

        Args:
            x (bs x state_dim x 1): state

        Returns:
            B (bs x state_dim x u_dim): B function for CCM
        """
        bs = x.shape[0]
        B = torch.zeros(bs, self.state_dim, self.u_dim).type(x.type())

        B[:, 4, 0] = 1 / self.mass
        B[:, 4, 1] = 1 / self.mass
        B[:, 5, 0] = self.l / self.J
        B[:, 5, 1] = -self.l / self.J

        return B