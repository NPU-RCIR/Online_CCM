'''
@Filename       : tracking_4_PVTOL.py
@Description    : 
@Create_Time    : 2025/02/20 10:39:09
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import random
import torch
import numpy as np
from sklearn.metrics import root_mean_squared_error as RMSE

from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from system.PVTOL import PVTOL
from online_learning.nn_based_disturbance_observer import nn_based_disturbance_observer, online_learning_disturbance

current_folder = os.getcwd()
dump_folder = os.path.join(current_folder,'dump')
fig_folder = os.path.join(current_folder,'fig')
log_folder = os.path.join(dump_folder)

# plt.rcParams.update({
#     "text.usetex": True
# })

font = fm.FontProperties(family='Times New Roman',size=12, stretch=0)
fontdict = {'family': 'Times New Roman',
            'size': 15
            }
plt.rc('font',family='Times New Roman') 
lw = 1.5

''' set seed variable for reproducible '''
seed_value = 2027
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  

torch.manual_seed(seed_value)     # random seed value for CPU
torch.cuda.manual_seed(seed_value)      # random seed value for GPU
torch.cuda.manual_seed_all(seed_value)   # random seed value for GPU (multi-GPUs)

model = PVTOL()

def tracking_nominal_trajectory(controller, ref_trajectory: np.array, simulation_steps):

    time_sequence = np.linspace(0,simulation_steps*model.delta_t,simulation_steps)

    tracking_trajectory = np.zeros([model.state_dim,simulation_steps])
    x0 = ref_trajectory[0,:model.state_dim,0]
    # xe0 = np.random.rand(model.state_dim) * 0.3
    # xe0[0] = random.uniform(0,0.6)
    # x0 = x0 + xe0
    xe0 = np.zeros(model.state_dim)
    xe0[1] = 0.1
    x0 = x0 + xe0

    for each_step in range(simulation_steps):
        
        tracking_trajectory[:model.state_dim,each_step] = x0       
        u = controller(torch.from_numpy(x0).reshape(1,model.state_dim,1), torch.from_numpy(x0 - ref_trajectory[0,:model.state_dim,each_step]).reshape(1,model.state_dim,1),
                        torch.from_numpy(ref_trajectory[0,model.state_dim:,each_step]).reshape(1,model.u_dim,1)).detach().cpu().numpy()[0] 

        ''' get the true state of the system '''
        # x0 = model.discrete_dynamics(model.delta_t*each_step,x0,u) 
        x0 = model.discrete_dynamics_disturbance(model.delta_t*each_step,x0,u) 
        # tracking_trajectory[model.state_dim:,each_step] = u
        
    return time_sequence, tracking_trajectory

def tracking_nominal_trajectory_online_learning(controller, ref_trajectory: np.array, simulation_steps):

    time_sequence = np.linspace(0,simulation_steps*model.delta_t,simulation_steps)

    tracking_trajectory = np.zeros([model.state_dim,simulation_steps])
    x0 = ref_trajectory[0,:model.state_dim,0]
    # xe0 = np.random.rand(model.state_dim) * 0.3
    # xe0[0] = random.uniform(0,0.6)
    # x0 = x0 + xe0
    xe0 = np.zeros(model.state_dim)
    xe0[1] = 0.1
    x0 = x0 + xe0

    true_disturbance = np.zeros([model.u_dim,simulation_steps])
    observation_disturbance = np.zeros([model.u_dim,simulation_steps])

    ''' nn based disturbance observer '''
    horizon = 10
    nn_observer = nn_based_disturbance_observer(input_dim=int(horizon/2)*model.state_dim, output_dim=model.u_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn_observer.parameters(), lr=0.001)
    epoch = 2
    
    ''' online data '''
    online_data = np.zeros([horizon,model.state_dim+model.u_dim])

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
            
            ''' online learning disturbance '''
            nn_observer = online_learning_disturbance(nn_observer=nn_observer, model=model, online_data=torch.from_numpy(online_data), criterion=criterion, optimizer=optimizer, epoch=epoch)

            ''' estimate disturbance '''
            disturbance_estimation = nn_observer(torch.from_numpy(online_data[int(horizon/2):,:model.state_dim].reshape(-1,int(horizon/2)*model.state_dim))).detach().numpy()

            observation_disturbance[0,each_step] = 1/model.mass*(disturbance_estimation[0,0] + disturbance_estimation[0,1])
            observation_disturbance[1,each_step] = model.l/model.J*(disturbance_estimation[0,0] - disturbance_estimation[0,1])
            true_disturbance[0,each_step] = 4*(np.cos(each_step*model.delta_t)+np.sin(x0[1])+np.cos(x0[3]))
            true_disturbance[1,each_step] = 2*(np.sin(x0[0])+np.cos(x0[4]))

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
        
    return time_sequence, tracking_trajectory, observation_disturbance, true_disturbance

def plot_4_tracking(time_sequence, simulation_steps, ref_trj, tracking_trajectory, tracking_trajectory_ol, fig_name):

    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(10,6))

    ax1.plot(time_sequence,ref_trj[0,0,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax1.plot(time_sequence,tracking_trajectory[0,0:simulation_steps],color=plt.cm.tab20c(1),label='Tracking Trajectory without OL',linewidth=lw)
    ax1.plot(time_sequence,tracking_trajectory_ol[0,0:simulation_steps],color=plt.cm.tab20b(2),label='Tracking Trajectory with OL',linewidth=lw)
    # ax1.set_xlabel('t [s]',fontsize=12)
    ax1.set_ylabel(r'X [m]',fontsize=12)
    ax1.set_xlim(0,simulation_steps*model.delta_t)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_sequence,ref_trj[0,1,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax2.plot(time_sequence,tracking_trajectory[1,0:simulation_steps],color=plt.cm.tab20c(1),label='Tracking Trajectory without OL',linewidth=lw)
    ax2.plot(time_sequence,tracking_trajectory_ol[1,0:simulation_steps],color=plt.cm.tab20b(2),label='Tracking Trajectory with OL',linewidth=lw)
    # ax2.set_xlabel('t [s]',fontsize=12)
    ax2.set_ylabel(r'Z [m]',fontsize=12)
    ax2.set_xlim(0,simulation_steps*model.delta_t)
    # ax2.legend()
    ax2.grid(True)

    ax3.plot(time_sequence,ref_trj[0,2,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax3.plot(time_sequence,tracking_trajectory[2,0:simulation_steps],color=plt.cm.tab20c(1),label='Tracking Trajectory without OL',linewidth=lw)
    ax3.plot(time_sequence,tracking_trajectory_ol[2,0:simulation_steps],color=plt.cm.tab20b(2),label='Tracking Trajectory with OL',linewidth=lw)
    # ax3.set_xlabel('t [s]',fontsize=12)
    ax3.set_ylabel(r'$\phi$ [rad]',fontsize=12)
    ax3.set_xlim(0,simulation_steps*model.delta_t)
    # ax3.legend()
    ax3.grid(True)

    ax4.plot(time_sequence,ref_trj[0,3,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax4.plot(time_sequence,tracking_trajectory[3,0:simulation_steps],color=plt.cm.tab20c(1),label='Tracking Trajectory without OL',linewidth=lw)
    ax4.plot(time_sequence,tracking_trajectory_ol[3,0:simulation_steps],color=plt.cm.tab20b(2),label='Tracking Trajectory with OL',linewidth=lw)
    # ax4.set_xlabel('t [s]',fontsize=12)
    ax4.set_ylabel(r'Vx [m/s]',fontsize=12)
    ax4.set_xlim(0,simulation_steps*model.delta_t)
    # ax4.legend()
    ax4.grid(True)

    ax5.plot(time_sequence,ref_trj[0,4,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax5.plot(time_sequence,tracking_trajectory[4,0:simulation_steps],color=plt.cm.tab20c(1),label='Tracking Trajectory without OL',linewidth=lw)
    ax5.plot(time_sequence,tracking_trajectory_ol[4,0:simulation_steps],color=plt.cm.tab20b(2),label='Tracking Trajectory with OL',linewidth=lw)
    ax5.set_xlabel('t [s]',fontsize=12)
    ax5.set_ylabel(r'Vz [m/s]',fontsize=12)
    ax5.set_xlim(0,simulation_steps*model.delta_t)
    # ax5.legend()
    ax5.grid(True)

    ax6.plot(time_sequence,ref_trj[0,5,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax6.plot(time_sequence,tracking_trajectory[5,0:simulation_steps],color=plt.cm.tab20c(1),label='Tracking Trajectory without OL',linewidth=lw)
    ax6.plot(time_sequence,tracking_trajectory_ol[5,0:simulation_steps],color=plt.cm.tab20b(2),label='Tracking Trajectory with OL',linewidth=lw)
    ax6.set_xlabel('t [s]',fontsize=12)
    ax6.set_ylabel(r'$\dot{\phi}$ [rad/s]',fontsize=12)
    ax6.set_xlim(0,simulation_steps*model.delta_t)
    # ax6.legend()
    ax6.grid(True)

    plt.savefig(os.path.join(fig_folder,fig_name),dpi=600,transparent=True,bbox_inches ='tight')

    plt.show()

def generate_tube(x, y, width):
    """
    生成围绕二维轨迹的带状区域边界点
    :param x: 轨迹的x坐标数组
    :param y: 轨迹的y坐标数组
    :param width: 带状区域的固定宽度
    :return: 上下边界的x和y坐标数组
    """
    # 计算切线方向（差分近似）
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # 归一化切线向量
    tangent = np.array([dx, dy])
    tangent /= np.linalg.norm(tangent, axis=0)
    
    # 计算法线方向（旋转90度）
    normal = np.array([-tangent[1], tangent[0]])  # 旋转矩阵 [0, -1; 1, 0]
    
    # 生成上下边界点
    upper_x = x + normal[0] * width / 2
    upper_y = y + normal[1] * width / 2
    lower_x = x - normal[0] * width / 2
    lower_y = y - normal[1] * width / 2
    
    return upper_x, upper_y, lower_x, lower_y

def plot_4_trajectory(time_sequence, simulation_steps, ref_trj, tracking_trajectory, tracking_trajectory_ol, fig_name):

    fig, ax = plt.subplots(figsize=(6,2))

    start_x = tracking_trajectory[0,0]
    start_y = tracking_trajectory[1,0]
    end_x = tracking_trajectory_ol[0,-1]
    end_y = tracking_trajectory_ol[1,-1]

    ref_x = ref_trj[0,0,:]
    ref_y = ref_trj[0,1,:]

    upper_x, upper_y, lower_x, lower_y = generate_tube(ref_x, ref_y, 0.25)
    ax.plot(upper_x, upper_y, '--', color=plt.cm.tab20c(2),alpha=0.8,linewidth=lw)
    ax.plot(lower_x, lower_y, '--', color=plt.cm.tab20c(2),alpha=0.8,linewidth=lw)
    ax.plot(start_x,start_y,'s',color=plt.cm.tab10(6),label='Start point')
    ax.plot(end_x,end_y,'p',color=plt.cm.tab10(4),label='End point')

    ax.plot(ref_trj[0,0,0:simulation_steps],ref_trj[0,1,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    ax.plot(tracking_trajectory[0,0:simulation_steps],tracking_trajectory[1,0:simulation_steps],':',color=plt.cm.tab20b(0),linewidth=lw)
    # ax.plot(tracking_trajectory[0,0:simulation_steps],tracking_trajectory[1,0:simulation_steps],':',color=plt.cm.tab20b(0),label='Tracking Trajectory without OL',linewidth=lw)
    ax.plot(tracking_trajectory_ol[0,0:simulation_steps],tracking_trajectory_ol[1,0:simulation_steps],color=plt.cm.tab20b(3),linewidth=lw)
    # ax.plot(tracking_trajectory_ol[0,0:simulation_steps],tracking_trajectory_ol[1,0:simulation_steps],color=plt.cm.tab20b(3),label='Tracking Trajectory with OL',linewidth=lw)

    axins = inset_axes(ax, width="15%", height="25%", loc='lower left',
                    bbox_to_anchor=(0.05, 0.18, 1, 1),
                    bbox_transform=ax.transAxes)
    
    axins.plot(ref_trj[0,0,0:simulation_steps],ref_trj[0,1,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    axins.plot(tracking_trajectory[0,0:simulation_steps],tracking_trajectory[1,0:simulation_steps],':',color=plt.cm.tab20b(0),linewidth=lw)
    axins.plot(tracking_trajectory_ol[0,0:simulation_steps],tracking_trajectory_ol[1,0:simulation_steps],color=plt.cm.tab20b(3),linewidth=lw)
    axins.plot(upper_x, upper_y, '--', color=plt.cm.tab20c(2),alpha=0.8,linewidth=lw)
    axins.plot(lower_x, lower_y, '--', color=plt.cm.tab20c(2),alpha=0.8,linewidth=lw)
    axins.set_xlim(3.5,4.5)
    axins.set_ylim(-0.7,-0.4)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.4)

    axins1 = inset_axes(ax, width="15%", height="25%", loc='lower left',
                    bbox_to_anchor=(0.27, 0.05, 1, 1),
                    bbox_transform=ax.transAxes)
    
    axins1.plot(ref_trj[0,0,0:simulation_steps],ref_trj[0,1,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    axins1.plot(tracking_trajectory[0,0:simulation_steps],tracking_trajectory[1,0:simulation_steps],':',color=plt.cm.tab20b(0),linewidth=lw)
    axins1.plot(tracking_trajectory_ol[0,0:simulation_steps],tracking_trajectory_ol[1,0:simulation_steps],color=plt.cm.tab20b(3),linewidth=lw)
    axins1.plot(upper_x, upper_y, '--', color=plt.cm.tab20c(2),alpha=0.8,linewidth=lw)
    axins1.plot(lower_x, lower_y, '--', color=plt.cm.tab20c(2),alpha=0.8,linewidth=lw)
    axins1.set_xlim(5.75,6.75)
    axins1.set_ylim(-1.25,-0.95)
    axins1.set_xticks([])
    axins1.set_yticks([])
    mark_inset(ax, axins1, loc1=1, loc2=2, fc="none", ec='k', lw=0.4)

    axins_ = inset_axes(ax, width="15%", height="25%", loc='lower left',
                    bbox_to_anchor=(0.4, 0.56, 1, 1),
                    bbox_transform=ax.transAxes)
    
    axins_.plot(ref_trj[0,0,0:simulation_steps],ref_trj[0,1,0:simulation_steps],'--',color=plt.cm.Paired(4),label='Reference Trajectory',linewidth=lw)
    axins_.plot(tracking_trajectory[0,0:simulation_steps],tracking_trajectory[1,0:simulation_steps],':',color=plt.cm.tab20b(0),linewidth=lw)
    axins_.plot(tracking_trajectory_ol[0,0:simulation_steps],tracking_trajectory_ol[1,0:simulation_steps],color=plt.cm.tab20b(3),linewidth=lw)
    axins_.plot(end_x,end_y,'p',color=plt.cm.tab10(4),label='End point')
    axins_.set_xlim(16.2,16.7)
    axins_.set_ylim(-2.1,-1.9)
    axins_.set_xticks([])
    axins_.set_yticks([])
    mark_inset(ax, axins_, loc1=3, loc2=4, fc="none", ec='k', lw=0.4)
    
    ax.set_xlabel('X [m]',fontsize=12)
    ax.set_ylabel('Z [m]',fontsize=12)
    ax.set_ylim(-2.5,0.25)
    ax.legend()
    ax.grid(True)

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

def tracking_4_PVTOL(args):
    simulation_steps = args.simulation_steps

    path = os.path.join(log_folder, args.experiment, args.model_name, args.extra)
    ccm_controller = torch.load(os.path.join(path, 'controller.pth'),map_location='cpu', weights_only=False)
    ccm_controller.eval()

    ref_path = os.path.join(log_folder, args.experiment)
    ref_x = np.load(os.path.join(ref_path, 'ref_x.npy'))
    ref_u = np.load(os.path.join(ref_path, 'ref_u.npy'))

    ref_trj = (np.vstack((ref_x,ref_u))).reshape(1,model.state_dim+model.u_dim,simulation_steps)

    ''' tracking globally (without re-planning locally) '''
    # with online learning
    # time_sequence, tracking_trajectory_ol, observation_disturbance, true_disturbance =  tracking_nominal_trajectory_online_learning(ccm_controller,ref_trj,simulation_steps)

    # # without online learning
    # time_sequence, tracking_trajectory =  tracking_nominal_trajectory(ccm_controller,ref_trj,simulation_steps)
    
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
    rmse = RMSE(ref_x,tracking_trajectory_ol)
    print('OL RMSE:{}'.format(rmse))
    rmse = RMSE(ref_x[:2,:],tracking_trajectory_ol[:2,:])
    print('VZ OL RMSE:{}'.format(rmse))
    rmse = RMSE(ref_x,tracking_trajectory)
    print('RMSE:{}'.format(rmse))
    rmse = RMSE(ref_x[:2,:],tracking_trajectory[:2,:])
    print('VZ RMSE:{}'.format(rmse))


    ''' plot '''
    # plot_4_tracking(time_sequence, simulation_steps, ref_trj, tracking_trajectory,tracking_trajectory_ol, 'state_PVTOL.png')
    # plot_disturbance(time_sequence,simulation_steps,observation_disturbance,true_disturbance, 'PVTOL_disturbance_error.png')
    plot_4_trajectory(time_sequence,simulation_steps,ref_trj,tracking_trajectory,tracking_trajectory_ol,'PVTOL_trajectory.png')