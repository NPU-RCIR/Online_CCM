'''
@Filename       : config.py
@Description    : 
@Create_Time    : 2025/02/17 10:45:55
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''


import argparse

parser = argparse.ArgumentParser()


'''general'''
parser.add_argument('--env', default='TSR',required=True, help='TSR | PVTOL')

'''parameters of dataset'''


'''configuration of tensorboard writer'''
parser.add_argument('--experiment', type=str, default='TSR', help='')
parser.add_argument('--model_name', type=str, default='online', help='')
parser.add_argument('--extra', type=str, default='', help='')

'''configuration of training'''
parser.add_argument('--bs', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')

''' configuration of evaluation '''
parser.add_argument('--eval_steps', type=int, help='evaluation steps')

def gen_args():
    args = parser.parse_args()

    if args.env == 'TSR':

        ''' for motion planning '''
        args.predictive_steps = 30
        args.simulation_steps = 2000
    
    if args.env == 'PVTOL':

        args.simulation_steps = 600

    return args