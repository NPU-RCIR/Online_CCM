'''
@Filename       : train.py
@Description    : 
@Create_Time    : 2025/02/17 17:31:45
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from system.TSR import TSR
from system.PVTOL import PVTOL
from ccm_nn.ccm_nn import CCM, controller_CCM, loss
from utils.training_utils.training_utils import create_writer

current_folder = os.getcwd()
dump_folder = os.path.join(current_folder,'dump')
data_folder = os.path.join(current_folder,'data')
log_folder = os.path.join(dump_folder)

from config import gen_args

device = "cuda" if torch.cuda.is_available() else "cpu"


def construct_dataset(model,
                      args,
                      train_num: int,
                      test_num: int):
    """construct dataset for training and testing

    Args:
        model (tether_model): model
        train_num (int): number of training examples
        test_num (int): number of testing examples

    Returns:
        dataset: dataset
    """
    if args.env == 'TSR':
        x_min = np.array([-0.8,    -0.99, -0.9 ,   0])
        x_max = np.array([0.1,      0.1,  -0.8,   0.6])

        u_ref_min = np.array([-3,-3])
        u_ref_max = np.array([3,3])

        lim = 0.05
        x_error_min = np.array([-lim,-lim,-lim,-lim])
        x_error_max = np.array([lim,lim,lim,lim])

    if args.env == 'PVTOL':
        x_min = np.array([-6.0, -0.2, -np.pi/3, -2, -1, -np.pi/3])
        x_max = np.array([6.0, 0.2, np.pi/3, 2, 1, np.pi/3])

        u_ref_min = np.array([model.mass*model.g/2-1,model.mass*model.g/2-1])
        u_ref_max = np.array([model.mass*model.g/2+1,model.mass*model.g/2+1])

        lim = 1.
        x_error_min = np.array([-lim,-lim,-lim,-lim,-lim,-lim])
        x_error_max = np.array([lim,lim,lim,lim,lim,lim])

    state_dim = model.state_dim
    u_dim = model.u_dim

    train_dataset = np.zeros((train_num,2*state_dim+u_dim,1))   # x, x_ref, u_ref
    test_dataset = np.zeros((test_num,2*state_dim+u_dim,1))

    for i in range(train_num):
        x_ref = (x_max-x_min)*np.random.rand(state_dim) + x_min # x_ref
        x_error = (x_error_max-x_error_min)*np.random.rand(state_dim) + x_error_min # x_error
        x = x_ref + x_error
        x[x>x_max] = x_max[x>x_max]
        x[x<x_min] = x_min[x<x_min]
        u_ref = (u_ref_max-u_ref_min)*np.random.rand(u_dim) + u_ref_min
        train_dataset[i,:state_dim,0] = x
        train_dataset[i,state_dim:2*state_dim,0] = x_ref
        train_dataset[i,2*state_dim:,0] = u_ref
    
    for i in range(test_num):
        x_ref = (x_max-x_min)*np.random.rand(state_dim) + x_min # x_ref
        x_error = (x_error_max-x_error_min)*np.random.rand(state_dim) + x_error_min # x_error
        x = x_ref + x_error
        x[x>x_max] = x_max[x>x_max]
        x[x<x_min] = x_min[x<x_min]
        u_ref = (u_ref_max-u_ref_min)*np.random.rand(u_dim) + u_ref_min
        test_dataset[i,:state_dim,0] = x
        test_dataset[i,state_dim:2*state_dim,0] = x_ref
        test_dataset[i,2*state_dim:,0] = u_ref
    
    return train_dataset, test_dataset


class training():
    def __init__(self,
                 state_dim: int,
                 u_dim: int,
                 epochs: int,
                 _lambda: float,
                 w_ub: float,
                 CCM: torch.nn.Module,
                 controller_CCM:torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 writer: torch.utils.tensorboard.writer.SummaryWriter,
                 log_dir: str,
                 loss_class
                 ):
        
        self.state_dim = state_dim
        self.u_dim = u_dim
        self.epochs = epochs

        self._lambda = _lambda
        self.w_ub = w_ub

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.CCM = CCM
        self.controller_CCM = controller_CCM
        self.optimizer = optimizer
        self.loss_class = loss_class
        self.writer = writer
        self.log_dir = log_dir
        self.lr_step = 10
    
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by every args.lr_step epochs

        Args:
            epoch (int): epoch
        """
        learning_rate = 1e-2
        lr = learning_rate * (0.1 ** (epoch // self.lr_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_step(self):
        """train step function

        Returns:
            numpy: loss of train step function
        """
        self.CCM.train()
        self.CCM.double()
        self.controller_CCM.train()
        self.controller_CCM.double()

        train_loss = 0
        train_p1 = 0
        train_p2 = 0
        train_l3 = 0

        pbar = tqdm(total=len(self.train_dataloader))

        for batch, trj in enumerate(self.train_dataloader):
            pbar.update(1)
            trj = trj.to(device)
            trj = trj.requires_grad_()
            
            loss,p1,p2,l3 = self.loss_class.ccm(trj,self._lambda, self.w_ub)
            train_loss += loss.item()
            train_p1 += p1.sum() / trj.shape[0]
            train_p2 += p2.sum() / trj.shape[0]
            train_l3 += l3 * trj.shape[0]
            
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
        pbar.close()

        return train_loss/len(self.train_dataloader), train_p1/len(self.train_dataloader), train_p2/len(self.train_dataloader), train_l3/len(self.train_dataloader)
        
    def test_step(self):
        """test step function

        Returns:
            numpy: loss of test step function
        """
        
        self.CCM.eval()
        self.CCM.double()
        self.controller_CCM.eval()
        self.controller_CCM.double()

        test_loss = 0
        test_p1 = 0
        test_p2 = 0
        test_l3 = 0

        for batch, trj in enumerate(self.test_dataloader):
            trj = trj.to(device)
            trj = trj.requires_grad_()
            
            loss,p1,p2,l3 = self.loss_class.ccm(trj, 0, self.w_ub)
            test_loss += loss.item()
            test_p1 += p1.sum() / trj.shape[0]
            test_p2 += p2.sum() / trj.shape[0]
            test_l3 += l3 * trj.shape[0]

        return test_loss/len(self.test_dataloader), test_p1/len(self.test_dataloader), test_p2/len(self.test_dataloader), test_l3/len(self.test_dataloader)
    

    def loop(self):
        """training loop

        Returns:
            results,best_loss: results,best_loss
        """

        best_loss = 0
        eval_steps = 1
        
        start_time = time.perf_counter()

        for epoch in range(self.epochs):
            # self.adjust_learning_rate(epoch=epoch)
            train_loss,p1,p2,l3 = self.train_step()
            
            self.writer.add_scalars(main_tag="train_Loss",
                                tag_scalar_dict={"train_loss":train_loss},
                                                global_step = epoch)

            if (epoch+1) %eval_steps == 0:
                test_loss,test_p1,test_p2,test_l3 = self.test_step()
                if (test_p1+test_p2) > best_loss:
                    best_loss = test_p1 + test_p2
                    #save the net including all parameters
                    torch.save(self.controller_CCM,(os.path.join(self.log_dir,"controller.pth")))
                    torch.save(self.CCM,(os.path.join(self.log_dir,"CCM.pth")))

                print(
                f"\r\nEpoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_p1: {test_p1:.4f} | "
                f"test_p2: {test_p2:.4f} | "
                f"test_l3: {test_l3:.4f} | "
                f"best_loss: {best_loss:.4f} | \r\n"
                )

                # # Update results dictionary
                # results["train_loss"].append(train_loss)
                # results["test_loss"].append(test_loss)

                # #writer.add_scalars('eval/test_loss',test_loss,epoch)
                # #writer.add_scalars('eval/best_loss',best_loss,epoch)

                self.writer.add_scalars(main_tag="eval_Loss",
                                tag_scalar_dict={"test_loss":test_loss,
                                                 "best_loss":best_loss},
                                                global_step = epoch)
        
        
        end_time = time.perf_counter()
        print("running time: {} seconds".format(end_time-start_time))
        print("best_loss: {}".format(best_loss))

        return best_loss

if __name__ == '__main__':

    args = gen_args()

    epochs = args.epochs
    batch_size = args.bs
    learning_rate = args.lr

    if args.env == 'TSR':
        model = TSR()
    if args.env == 'PVTOL':
        model = PVTOL()

    train_dataset, test_dataset = construct_dataset(model, args, 131072, 32768)
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 #num_workers=using_cores,
                                 shuffle=False)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                #num_workers=using_cores
                                )

    CCM_net = CCM(state_dim=model.state_dim, u_dim=model.u_dim, w_lb=0.1).to(device)
    controller_CCM_net = controller_CCM(state_dim=model.state_dim, u_dim=model.u_dim).to(device)
    # CCM_net = torch.compile(CCM_net) #for pytorch2.0
    # controller_CCM_net = torch.compile(controller_CCM_net)
    # loss_func = nn.MSELoss()
    loss_ = loss(CCM_net,controller_CCM_net,model,device)

    optimizer = torch.optim.Adam(list(CCM_net.parameters())+list(controller_CCM_net.parameters()), lr=learning_rate)

    writer, log_dir = create_writer(folder=log_folder, experiment_name=args.env, model_name=args.model_name, extra=args.extra)
    
    training_instance = training(state_dim = model.state_dim, 
                                 u_dim = model.u_dim, 
                                 epochs=epochs, 
                                 _lambda=0.5, 
                                 w_ub=10, 
                                 CCM=CCM_net, 
                                 controller_CCM=controller_CCM_net, 
                                 train_dataloader=train_dataloader, 
                                 test_dataloader=test_dataloader, 
                                 optimizer=optimizer,
                                 writer=writer,
                                 log_dir=log_dir,
                                 loss_class=loss_)

    best_loss = training_instance.loop()