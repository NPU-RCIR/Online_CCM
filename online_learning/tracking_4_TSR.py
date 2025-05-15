'''
@Filename       : tracking_4_TSR.py
@Description    : 
@Create_Time    : 2025/02/18 21:54:50
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import random
import time
import torch
import numpy as np
from sklearn.metrics import root_mean_squared_error as RMSE

from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from system.TSR import TSR
from online_learning.nn_based_disturbance_observer import nn_based_disturbance_observer, online_learning_disturbance

current_folder = os.getcwd()
dump_folder = os.path.join(current_folder,'dump')
fig_folder = os.path.join(current_folder,'fig')
log_folder = os.path.join(dump_folder)

font = fm.FontProperties(family='Times New Roman',size=12, stretch=0)
fontdict = {'family': 'Times New Roman',
            'size': 15
            }
plt.rc('font',family='Times New Roman') 
lw = 1.5

''' set seed variable for reproducible '''
seed_value = 9357
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  

torch.manual_seed(seed_value)     # random seed value for CPU
torch.cuda.manual_seed(seed_value)      # random seed value for GPU
torch.cuda.manual_seed_all(seed_value)   # random seed value for GPU (multi-GPUs)

model = TSR()
disturbance_upper_bound = 0.5
disturbance_lower_bound = -0.5

def tracking_nominal_trajectory(controller, ref_trajectory: np.array, simulation_steps):

    time_sequence = np.linspace(0,simulation_steps*model.delta_t,simulation_steps)

    tracking_trajectory = np.zeros([model.state_dim,simulation_steps])
    x0 = ref_trajectory[0,:model.state_dim,0]
    xe0 = np.random.rand(model.state_dim) * 0.010
    x0 = x0 + xe0

    for each_step in range(simulation_steps):
        
        tracking_trajectory[:model.state_dim,each_step] = x0       
        u = controller(torch.from_numpy(x0).reshape(1,model.state_dim,1), torch.from_numpy(x0 - ref_trajectory[0,:model.state_dim,each_step]).reshape(1,model.state_dim,1),
                        torch.from_numpy(ref_trajectory[0,model.state_dim:,each_step]).reshape(1,model.u_dim,1)).detach().cpu().numpy()[0] 

        ''' get the true state of the system '''
        x0 = model.discrete_dynamics_disturbance(model.delta_t*each_step,x0,u) 
        # tracking_trajectory[model.state_dim:,each_step] = u
        
    return time_sequence, tracking_trajectory

def tracking_nominal_trajectory_online_learning(controller, ref_trajectory: np.array, simulation_steps):

    time_sequence = np.linspace(0,simulation_steps*model.delta_t,simulation_steps)

    tracking_trajectory = np.zeros([model.state_dim,simulation_steps])
    x0 = ref_trajectory[0,:model.state_dim,0]
    xe0 = np.random.rand(model.state_dim) * 0.010
    x0 = x0 + xe0

    true_disturbance = np.zeros([model.u_dim,simulation_steps])
    observation_disturbance = np.zeros([model.u_dim,simulation_steps])

    ''' nn based disturbance observer '''
    horizon = 40
    nn_observer = nn_based_disturbance_observer(input_dim=int(horizon/2)*model.state_dim, output_dim=model.u_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn_observer.parameters(), lr=0.001)
    epoch = 2
    
    ''' online data (memory buffer) '''
    online_data = np.zeros([horizon,model.state_dim+model.u_dim])

    # computation_time = []

    for each_step in range(simulation_steps):
        
        if each_step < horizon:
            ''' simulate disturbance dynamics '''
            tracking_trajectory[:model.state_dim,each_step] = x0       
            u = controller(torch.from_numpy(x0).reshape(1,model.state_dim,1), torch.from_numpy(x0 - ref_trajectory[0,:model.state_dim,each_step]).reshape(1,model.state_dim,1),
                            torch.from_numpy(ref_trajectory[0,model.state_dim:,each_step]).reshape(1,model.u_dim,1)).detach().cpu().numpy()[0][:,0] 

            ''' get the true state of the system '''
            x0 = model.discrete_dynamics_disturbance(model.delta_t*each_step,x0,u) 

            ''' construct online data '''
            online_data[each_step,:] = np.hstack((x0,u))
        else:
            # start_time = time.perf_counter()

            ''' online learning disturbance '''
            nn_observer = online_learning_disturbance(nn_observer=nn_observer, model=model, online_data=torch.from_numpy(online_data), criterion=criterion, optimizer=optimizer, epoch=epoch)
            # end_time = time.perf_counter()
            # computation_time.append(end_time-start_time)

            ''' estimate disturbance '''
            disturbance_estimation = nn_observer(torch.from_numpy(online_data[int(horizon/2):,:model.state_dim].reshape(-1,int(horizon/2)*model.state_dim))).detach().numpy()

            ''' clip '''
            disturbance_estimation = np.clip(disturbance_estimation,a_min=disturbance_lower_bound, a_max=disturbance_upper_bound)

            ''' record the estimation of unknown external disturbance '''
            observation_disturbance[:,each_step] = disturbance_estimation
            true_disturbance[0,each_step] = 0.3*(np.cos(each_step*model.delta_t)+np.sin(x0[2]))
            true_disturbance[1,each_step] = 0.2*(np.cos(x0[1]))

            ''' simulate the system with disturbance '''
            tracking_trajectory[:model.state_dim,each_step] = x0       
            u = controller(torch.from_numpy(x0).reshape(1,model.state_dim,1), torch.from_numpy(x0 - ref_trajectory[0,:model.state_dim,each_step]).reshape(1,model.state_dim,1),
                            torch.from_numpy(ref_trajectory[0,model.state_dim:,each_step]).reshape(1,model.u_dim,1)).detach().cpu().numpy()[0][:,0] 
            
            u = u - disturbance_estimation[0,:]

            ''' get the true state of the system '''
            x0 = model.discrete_dynamics_disturbance(model.delta_t*each_step,x0,u) 

            ''' construct online data '''
            online_data = np.append(online_data, np.hstack((x0,u)).reshape(-1,model.state_dim+model.u_dim), axis=0)
            online_data = np.delete(online_data,0,axis=0)

    # print('computation time mean:{}, std:{}'.format(np.mean(computation_time)*1000,np.std(computation_time)*1000))

    return time_sequence, tracking_trajectory, observation_disturbance, true_disturbance

def plot_4_tracking(time_sequence, simulation_steps, ref_trj, tracking_trajectory, tracking_trajectory_ol, fig_name):

    precision_line_upper = np.linspace(0.01,0.01,simulation_steps)
    precision_line_lower = np.linspace(-0.01,-0.01,simulation_steps)

    fig,((ax1,ax2)) = plt.subplots(1,2,figsize=(8.5, 3))
    ax1.plot(time_sequence,ref_trj[0,0,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax1.plot(time_sequence,tracking_trajectory[0,0:simulation_steps],color=plt.cm.tab20c(1),label='Tracking Trajectory without OL',linewidth=lw)
    ax1.plot(time_sequence,tracking_trajectory_ol[0,0:simulation_steps],color=plt.cm.tab20b(2),label='Tracking Trajectory with OL',linewidth=lw)
    ax1.plot(time_sequence,precision_line_upper,'-.',color=plt.cm.tab20c(16),linewidth=1,label='1% Precision')
    ax1.plot(time_sequence,precision_line_lower,'-.',color=plt.cm.tab20c(16),linewidth=1,label='-1% Precision')
    ax1.set_xlabel('True anomaly [rad]',fontsize=12)
    ax1.set_ylabel(r'Inplane Angle [rad]',fontsize=12)
    ax1.set_xlim(0,simulation_steps*0.01)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_sequence,(ref_trj[0,1,0:simulation_steps]),'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax2.plot(time_sequence,(tracking_trajectory[1,0:simulation_steps]),color=plt.cm.tab20c(1),label='Tracking Trajectory without OL',linewidth=lw)
    ax2.plot(time_sequence,(tracking_trajectory_ol[1,0:simulation_steps]),color=plt.cm.tab20b(2),label='Tracking Trajectory with OL',linewidth=lw)
    ax2.plot(time_sequence,precision_line_upper,'-.',color=plt.cm.tab20c(16),linewidth=1,label='1% Precision')
    ax2.plot(time_sequence,precision_line_lower,'-.',color=plt.cm.tab20c(16),linewidth=1,label='-1% Precision')
    ax2.set_xlabel('True anomaly [rad]',fontsize=12)
    ax2.set_ylabel(r'Dimensionless Tether Length',fontsize=12)
    ax2.set_xlim(0,simulation_steps*0.01)
    ax2.legend()
    ax2.grid(True)

    plt.savefig(os.path.join(fig_folder,fig_name),dpi=600,transparent=True,bbox_inches ='tight')

    plt.show() 

def plot_disturbance(time_sequence, simulation_steps,observation_disturbance, true_disturbance, fig_name):

    fig,((ax1,ax2)) = plt.subplots(1,2,figsize=(8.5, 3))
    ax1.plot(time_sequence,true_disturbance[0,0:simulation_steps],'--',color=plt.cm.Paired(4),label='True',linewidth=lw)
    ax1.plot(time_sequence,observation_disturbance[0,0:simulation_steps],color=plt.cm.Paired(1),label='Esimation',linewidth=lw)

    ax2.plot(time_sequence,true_disturbance[1,0:simulation_steps],'--',color=plt.cm.Paired(4),label='True',linewidth=lw)
    ax2.plot(time_sequence,observation_disturbance[1,0:simulation_steps],color=plt.cm.Paired(1),label='Esimation',linewidth=lw)
    ax2.legend()
    ax2.grid(True)

    fig,ax1 = plt.subplots(figsize=(5,2.5))
    ax1.plot(time_sequence[40:],np.linalg.norm(observation_disturbance-true_disturbance,axis=0)[40:]/2,color='k',linewidth=lw)
    # ax1.plot(time_sequence[40:],np.linalg.norm(true_disturbance,axis=0)[40:],color='r',linewidth=lw)
    ax1.set_xlabel("True anomaly [rad]",fontsize=12)
    ax1.set_ylabel("Prediction Error of Uncertainty",fontsize=12)
    ax1.grid(True)

    plt.savefig(os.path.join(fig_folder,fig_name),dpi=600,transparent=True,bbox_inches ='tight')

    plt.show()

def tracking_4_TSR(args):
    simulation_steps = args.simulation_steps

    path = os.path.join(log_folder, args.experiment, args.model_name, args.extra)
    ccm_controller = torch.load(os.path.join(path, 'controller.pth'),map_location='cpu', weights_only=False)
    ccm_controller.eval()

    ref_x = np.load(os.path.join(path, 'ref_x.npy'))
    ref_u = np.load(os.path.join(path, 'ref_u.npy'))

    ref_trj = (np.vstack((ref_x,ref_u))).reshape(1,model.state_dim+model.u_dim,simulation_steps)


    ''' tracking globally (without re-planning locally) '''
    # # without online learning
    # time_sequence, tracking_trajectory =  tracking_nominal_trajectory(ccm_controller,ref_trj,simulation_steps)

    # # with online learning
    # time_sequence, tracking_trajectory_ol, observation_disturbance, true_disturbance =  tracking_nominal_trajectory_online_learning(ccm_controller,ref_trj,simulation_steps)

    path_save_results = os.path.join(path,'results')


    ''' save results '''
    # np.save(os.path.join(path_save_results,'tracking_trajectory.npy'),tracking_trajectory)
    # np.save(os.path.join(path_save_results,'tracking_trajectory_ol.npy'),tracking_trajectory_ol)
    # np.save(os.path.join(path_save_results,'observation_disturbance.npy'),observation_disturbance)
    # np.save(os.path.join(path_save_results,'true_disturbance.npy'),true_disturbance)
    # np.save(os.path.join(path_save_results,'time_sequence.npy'),time_sequence)


    ''' load results '''
    tracking_trajectory = np.load(os.path.join(path_save_results,'tracking_trajectory.npy'))
    tracking_trajectory_ol = np.load(os.path.join(path_save_results,'tracking_trajectory_ol.npy'))
    observation_disturbance = np.load(os.path.join(path_save_results,'observation_disturbance.npy'))
    true_disturbance = np.load(os.path.join(path_save_results,'true_disturbance.npy'))
    time_sequence = np.load(os.path.join(path_save_results,'time_sequence.npy'))


    ''' RMSEs '''
    rmse = RMSE(ref_x,tracking_trajectory)
    print('RMSE:{}'.format(rmse))
    rmse = RMSE(ref_x,tracking_trajectory_ol)
    print('OL RMSE:{}'.format(rmse))


    ''' plot '''
    plot_4_tracking(time_sequence, simulation_steps, ref_trj, tracking_trajectory,  tracking_trajectory_ol, 'state_TSR.png')
    plot_disturbance(time_sequence,simulation_steps,observation_disturbance,true_disturbance, 'TSR_disturbance_error.png')