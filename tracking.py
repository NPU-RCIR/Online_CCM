'''
@Filename       : tracking.py
@Description    : 
@Create_Time    : 2025/02/18 09:51:14
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import numpy as np

from config import gen_args
from online_learning import tracking_4_TSR, tracking_4_PVTOL

current_folder = os.getcwd()
dump_folder = os.path.join(current_folder,'dump')
log_folder = os.path.join(dump_folder)

if __name__ == '__main__':

    args = gen_args()

    if args.env == 'TSR':
        tracking_4_TSR.tracking_4_TSR(args)

    if args.env == 'PVTOL':
        tracking_4_PVTOL.tracking_4_PVTOL(args)