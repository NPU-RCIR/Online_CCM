import importlib
import numpy as np
from motion_planning.PVTOL.utils import EulerIntegrate
from motion_planning.PVTOL.np2pth import get_system_wrapper
from motion_planning.PVTOL.configs import config_PVTOL as config
from matplotlib import pyplot as plt

import sys
import os
sys.path.append('motion_planning/PVTOL/systems')
sys.path.append('motion_planning/PVTOL/configs')
sys.path.append('motion_planning/PVTOL/models')


def planning_4_PVTOL(save_path=None):

    system = importlib.import_module('system_PVTOL')
    f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)

    time_bound = config.time_bound
    time_step = config.time_step

    x_0, xstar_0, ustar = config.system_reset(np.random.rand())
    ustar = [u.reshape(-1,1) for u in ustar]
    xstar_0 = xstar_0.reshape(-1,1)
    xstar, _ = EulerIntegrate(None, f, B, None, ustar, xstar_0, time_bound, time_step, with_tracking=False)
    # print(x_0.shape,xstar_0.shape,ustar)

    xstar = np.array(xstar).squeeze(axis=2)
    ustar = np.array(ustar).squeeze(axis=2)

    # print(xstar[1:,:].shape, ustar.shape)

    # plt.plot(xstar[:,0], xstar[:,1])
    # plt.show()

    if save_path is not None:
        np.save(os.path.join(save_path,"ref_x"), xstar[1:,].transpose())
        np.save(os.path.join(save_path,"ref_u"), ustar.transpose())