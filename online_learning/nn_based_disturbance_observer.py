'''
@Filename       : nn_based_disturbance_observer.py
@Description    : 
@Create_Time    : 2025/02/18 22:29:57
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np

class nn_based_disturbance_observer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(nn_based_disturbance_observer, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, input_):
        return self.net(input_)

def online_learning_disturbance(nn_observer, model, online_data, criterion, optimizer, epoch):
    """online learning disturbance from historical data

    Args:
        nn_observer (nn.module): neural network based disturbance observer
        model (model class): model
        online_data (tensor T x (state_dim+u_dim)): _description_
        criterion (nn.Module): criterion
        optimizer (nn.Module): optimizer
        epoch (int): training epoch
    """
    
    nn_observer.train()
    nn_observer.double()

    horizon = int(online_data.shape[0]/2)
    scaling = 20

    for _ in range(epoch):
        predict_state = torch.zeros(horizon, model.state_dim).type(online_data.type())
        optimizer.zero_grad()
        for i in range(horizon-1):
            
            ''' construct the input of nn_based_disturbance_observer '''
            input_ = online_data[i:(i+horizon),:model.state_dim].reshape(-1,horizon*model.state_dim)
            
            ''' estimate disturbance '''
            disturbance_estimation  = nn_observer(input_)[0,:]
            
            ''' compensate the estimation into control channel '''
            control_ = online_data[i+horizon-1,model.state_dim:] + disturbance_estimation

            ''' simulate the system with nominal dynamics '''
            predict_state[i+1,:] = model.discrete_dynamics(0, online_data[i+horizon-1,:model.state_dim], control_)

        loss = scaling*criterion(predict_state, online_data[horizon:,:model.state_dim])
        loss.backward()
        optimizer.step()

    return nn_observer