'''
@Filename       : planning.py
@Description    : 
@Create_Time    : 2025/02/17 21:42:01
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import numpy as np

from config import gen_args
from motion_planning.planning_4_TSR import planning_4_TSR, plot_4_TSR
from motion_planning.planning_4_PVTOL import planning_4_PVTOL

current_folder = os.getcwd()
dump_folder = os.path.join(current_folder,'dump')
log_folder = os.path.join(dump_folder)


if __name__ == "__main__":
    args = gen_args()

    save_path = os.path.join(log_folder, args.experiment, args.model_name, args.extra)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if args.env == 'TSR':
        predictive_steps = args.predictive_steps
        simulation_steps = args.simulation_steps
        time_sequence, state, control = planning_4_TSR(predictive_steps=predictive_steps, simulation_steps=simulation_steps, save_path=save_path)
        # plot_4_TSR(time_sequence,state,control)
    
    if args.env == 'PVTOL':
        planning_4_PVTOL(save_path=save_path)