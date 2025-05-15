'''
@Filename       : planning_4_TSR.py
@Description    : 
@Create_Time    : 2025/02/17 21:28:17
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import casadi as cas
import numpy as np
from tqdm import tqdm

from system import TSR

import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
font = fm.FontProperties(family='Times New Roman',size=12, stretch=0)
fontdict = {'family': 'Times New Roman',
            'size': 15
            }
plt.rc('font',family='Times New Roman') 
lw = 1.5

def planning_4_TSR(predictive_steps, simulation_steps, save_path=None):
    model = TSR.TSR()
    N = predictive_steps
    state_dim = model.state_dim
    control_dim = model.u_dim

    U = cas.SX.sym("U", control_dim, N)  # control in N steps
    X = cas.SX.sym("X", state_dim, N + 1)  # state of system in N+1 steps(as there is n steps control, the dimension of X is state_dim*(N+1))
    P = cas.SX.sym("P", state_dim + state_dim)  # parameter matrix that will be passed to optimizer, P:[x0,xf]

    X[:, 0] = P[:state_dim]

    for i in range(N):
        X[:, i + 1] = model.discrete_dynamics_casadi(0, X[:, i], U[:, i])

    # state of system in N prediction steps
    discrete_model = cas.Function(
        "discrete_model", [U, P], [X], ["input_u", "target_state"], ["states"]
    )

    Q = 1 * np.eye(state_dim, state_dim)
    R = 0.5 * np.eye(control_dim, control_dim)

    # cost function
    obj = 0
    for i in range(N):
        obj += cas.mtimes(
            [(X[:, i] - P[state_dim:]).T, Q, (X[:, i] - P[state_dim:])]
        ) + cas.mtimes([U[:, i].T, R, U[:, i]])
    obj += cas.mtimes([(X[:, N] - P[state_dim:]).T, Q, (X[:, N] - P[state_dim:])])

    # obstacle_l = 0.5
    # obstacle_theta = 5*np.pi/6
    # obstacle_x = obstacle_l*np.sin(obstacle_theta)
    # obstacle_y = obstacle_l*np.cos(obstacle_theta)
    obstacle_x = 0.15
    obstacle_y = -0.6

    g = []
    for i in range(N + 1):
        x_ = (X[1,i])*cas.sin(X[0,i])
        y_ = (X[1,i])*cas.cos(X[0,i])
        d = cas.sqrt((x_-obstacle_x)**2+(y_-obstacle_y)**2)
        g.append(d)
        # g.append(X[0, i])
        # g.append(X[1, i])
        g.append(X[1, i])
        g.append(X[3, i])

    nlp_prob = {
        "f": obj,
        "x": cas.reshape(U, -1, 1),
        "p": P,
        "g": cas.vertcat(*g),
    }  # g是一维向量
    opts_setting = {
        "ipopt.max_iter": 100,
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.acceptable_obj_change_tol": 1e-6,
    }

    solver = cas.nlpsol("solver", "ipopt", nlp_prob, opts_setting)

    # state constrains
    lbg = []
    ubg = []
    # control input constrains
    lbx = []
    ubx = []

    for _ in range(N + 1):
        lbg.append(0.06)
        ubg.append(100)
        # lbg.append(-0.6)
        # ubg.append(0.2)
        # lbg.append(-1)
        # ubg.append(1)
        lbg.append(-0.99)
        ubg.append(0.0)
        lbg.append(0)
        ubg.append(3)

    for _ in range(N):
        lbx.append(-3)
        ubx.append(3)
        lbx.append(-3)
        ubx.append(3)

    t0 = 0.0
    x0 = np.array([0, -0.99, 0, 2.5]).reshape(-1, 1)  # initial state
    xs = np.array([0, 0, 0, 0]).reshape(-1, 1)  # final state
    u0 = np.array([0,0] * N).reshape(control_dim, N)  # initial u0

    h = model.delta_t
    sim_time = simulation_steps * h

    mpciter = 0
    pbar = tqdm(total=simulation_steps)

    x = np.zeros((state_dim, simulation_steps))
    control = np.zeros((control_dim, simulation_steps))
    time_sequence = np.linspace(0, simulation_steps * 0.01, simulation_steps)

    # while np.linalg.norm(x0 - xs) > 1e-4 and mpciter - sim_time / h < 0.0:
    while  mpciter - sim_time / h < 0.0:
        c_p = np.concatenate([x0, xs])

        init_control = cas.reshape(u0, -1, 1)
        x[:, mpciter] = x0[:, 0]
        control[0, mpciter] = u0[0, 0]
        control[1, mpciter] = u0[1, 0]

        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)

        u_sol = cas.reshape(res["x"], control_dim, N)

        ff_value = discrete_model(
            u_sol, c_p
        )  # get the state of system in N prediction steps

        x0 = (model.discrete_dynamics(0, x0[:, 0], u_sol[:, 0])).reshape(-1, 1)
        u0 = u_sol

        mpciter += 1
        pbar.update(1)
    pbar.close()

    if save_path is not None:
        np.save(os.path.join(save_path,"ref_x"), x)
        np.save(os.path.join(save_path,"ref_u"), control)

    return time_sequence, x, control

def plot_4_TSR(time_sequence,x,control):
    simulation_steps = len(time_sequence)

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 7))
    # plt.subplots_adjust(
    #     left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.6
    # )

    ax1.plot(time_sequence, x[0, :], linewidth=lw)
    ax1.grid(True)
    ax1.set_xlabel("True anomaly[rad]", fontsize=12)
    ax1.set_ylabel(r"Inplane Angle [rad]", fontsize=12)
    ax1.set_xlim(0, simulation_steps * 0.01)
    # ax1.set_title(r"$\theta$")

    ax2.plot(time_sequence, x[2, :], linewidth=lw)
    ax2.grid(True)
    ax2.set_xlabel("True anomaly[rad]", fontsize=12)
    ax2.set_ylabel(r"Velocity of Inplane Angle[rad/s]", fontsize=12)
    # ax2.set_title(r"$\dot \theta$")
    ax2.set_xlim(0, simulation_steps * 0.01)

    ax3.plot(time_sequence, x[1, :], linewidth=lw)
    ax3.grid(True)
    ax3.set_xlabel("True anomaly[rad]", fontsize=12)
    ax3.set_ylabel(r"Dimensionless Tehter Length", fontsize=12)
    # ax3.set_title()
    ax3.set_xlim(0, simulation_steps * 0.01)

    ax4.plot(time_sequence, x[3, :], linewidth=lw)
    ax4.grid(True)
    ax4.set_xlabel("True anomaly[rad]", fontsize=12)
    ax4.set_ylabel(r"Dimensionless Velocity of Tehter", fontsize=12)
    ax4.set_xlim(0, simulation_steps * 0.01)
    # ax4.set_title(r"$\dot l$")

    # plt.savefig("state.png", dpi=600, transparent=True, bbox_inches="tight")

    fig, ((ax1),(ax2)) = plt.subplots(2,1)
    ax1.plot(time_sequence, control[0, :], linewidth=lw)
    ax1.grid(True)
    # ax1.set_xlabel("True anomaly(rad)", fontsize=12)
    ax1.set_ylabel(r"Control Input", fontsize=12)
    ax1.set_xlim(0, simulation_steps * 0.01)

    ax2.plot(time_sequence, control[1, :], linewidth=lw)
    ax2.grid(True)
    ax2.set_xlabel("True anomaly[rad]", fontsize=12)
    ax2.set_ylabel(r"Control Input", fontsize=12)
    ax2.set_xlim(0, simulation_steps * 0.01)

    # plt.savefig("tension.png", dpi=600, transparent=True, bbox_inches="tight")

    length = x[1,:]
    theta = x[0,:]
    x = length*np.sin(theta)
    y = length*np.cos(theta)
    start_point_x = x[0]
    start_point_y = y[0]
    end_point_x = x[-1]
    end_point_y = y[-1]

    # obstacle_l = 0.5
    # obstacle_theta = 5*np.pi/6
    # obstacle_x = obstacle_l*np.sin(obstacle_theta)
    # obstacle_y = obstacle_l*np.cos(obstacle_theta)
    obstacle_x = 0.15
    obstacle_y = -0.6
    r = 0.06
    a_x = np.arange(0,2*np.pi,0.01)
    x_ = obstacle_x + r*np.cos(a_x)
    y_ = obstacle_y + r*np.sin(a_x)

    fig,(ax) = plt.subplots(figsize=(5, 8))
    ax.plot(x,y)
    ax.plot(obstacle_x,obstacle_y,'^',label="obstacle")
    ax.plot(start_point_x,start_point_y,'s',label='Start point')
    ax.plot(end_point_x,end_point_y,'p',label='End point')
    ax.plot(x_,y_,'-',linewidth=lw)
    ax.set_xlabel('X',fontsize=12)
    ax.set_ylabel('Y',fontsize=12)
    ax.legend()
    ax.grid(True)

    plt.axis("equal")
    plt.show()