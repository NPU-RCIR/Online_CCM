'''
@Filename       : ccm_nn.py
@Description    : 
@Create_Time    : 2025/02/17 10:46:12
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import torch
from torch import nn
from ccm_nn.jacobian_matrix import batched_jacobian, Bbot_func, weighted_gradients

device = "cuda" if torch.cuda.is_available() else "cpu"

class CCM(nn.Module):
    """
    Control Contrction Metrics class
    """
    def __init__(self, state_dim, u_dim, w_lb):
        super(CCM, self).__init__()
        self.state_dim = state_dim
        self.u_dim = u_dim
        self.w_lb = w_lb
        # dim = state_dim - u_dim
        dim = state_dim

        self.W = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(), 
            nn.Linear(128, state_dim * state_dim, bias=False))
        
        self.W_bot = nn.Sequential(
            nn.Linear(dim - u_dim, 128),
            nn.Tanh(),
            nn.Linear(128, (state_dim - u_dim)**2, bias=False))
        
    
    def forward(self, x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        W = self.W(x[:, :self.state_dim]).view(bs, self.state_dim, self.state_dim)
        Wbot = self.W_bot(x[:, :self.state_dim-self.u_dim]).view(bs, self.state_dim-self.u_dim, self.state_dim-self.u_dim)
        W[:, 0:self.state_dim-self.u_dim, 0:self.state_dim-self.u_dim] = Wbot
        W[:, self.state_dim-self.u_dim::, 0:self.state_dim-self.u_dim] = 0

        W = W.transpose(1,2).matmul(W)
        W = W + self.w_lb * torch.eye(self.state_dim).view(1, self.state_dim, self.state_dim).type(x.type())
        return W

class controller_CCM(nn.Module):
    """
    Controller class
    """
    def __init__(self, state_dim, u_dim):
        super(controller_CCM, self).__init__()
        self.state_dim = state_dim
        self.u_dim = u_dim

        dim = state_dim - u_dim
        c = 3 * state_dim

        self.w_1 = nn.Sequential(
            nn.Linear(2 * dim, 128),
            nn.Tanh(), 
            nn.Linear(128, c * state_dim))
        
        self.w_2 = nn.Sequential(
            nn.Linear(2 * dim, 128),
            nn.Tanh(),
            nn.Linear(128, u_dim * c))
    
    def forward(self, x, x_error, u_ref):
        bs = x.shape[0]

        w1 = self.w_1(torch.cat([x[:,self.u_dim:self.state_dim,:],(x-x_error)[:,self.u_dim:self.state_dim,:]],dim=1).squeeze(-1)).reshape(bs, -1, self.state_dim)
        w2 = self.w_2(torch.cat([x[:,self.u_dim:self.state_dim,:],(x-x_error)[:,self.u_dim:self.state_dim,:]],dim=1).squeeze(-1)).reshape(bs, self.u_dim, -1)
        u = w2.matmul(torch.tanh(w1.matmul(x_error))) + u_ref

        return u

def positive_defined_matrix(A):
    """Give penalty for negative defined matrices

    Args:
        A (tensor): bs x d x d

    Returns:
        float: penalty
    """
    K = 1024
    z = torch.randn(K, A.size(-1), dtype = torch.float64).to(device) # z: K x d
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1,K,-1)).sum(dim=2).view(-1)
    negative_index = zTAz.detach().cpu().numpy() < 0
    if negative_index.sum()>0:
        negative_zTAz = zTAz[negative_index]
        return -1.0 * (negative_zTAz.mean())
    else:
        return torch.tensor(0.).type(z.type()).requires_grad_()   


class loss():
    def __init__(self,
                 CCM: torch.nn.Module,
                 controller_CCM: torch.nn.Module,
                 model,
                 device):
        self.CCM = CCM
        self.controller_CCM = controller_CCM
        self.f = model.model4CCM_f
        self.B = model.model4CCM_B
        self.state_dim = model.state_dim
        self.u_dim = model.u_dim
        self.device = device
      
    # dataset: x, x_ref, u_ref
    def ccm(self,
            dataset,
            _lambda,
            w_ub):
        
        bs = dataset.shape[0]
        x = dataset[:,:self.state_dim]
        x_ref = dataset[:,self.state_dim:2*self.state_dim]
        u_ref = dataset[:,2*self.state_dim:]

        W = self.CCM(x)
        M = torch.inverse(W)
        f = self.f(x).to(self.device)
        B = self.B(x).to(self.device)

        DfDx = batched_jacobian(f,x)

        DBDx = torch.zeros(bs, self.state_dim, self.state_dim, self.u_dim).type(x.type())
        for i in range(self.u_dim):
            DBDx[:,:,:,i] = batched_jacobian(B[:,:,i].unsqueeze(-1), x)

        _Bbot = Bbot_func(x,self.state_dim,self.u_dim).to(self.device)
        u = self.controller_CCM(x, x - x_ref, u_ref) # u: bs x m x 1 
        K = batched_jacobian(u, x)

        A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(self.u_dim)])
        dot_x = f + B.matmul(u)
        dot_M = weighted_gradients(M, dot_x, x, detach=False) # DMDt
        dot_W = weighted_gradients(W, dot_x, x, detach=False) # DWDt

        Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M) + M.matmul(A + B.matmul(K)) + 2 * _lambda * M
        
        # C1
        C1_inner = - weighted_gradients(W, f, x) + DfDx.matmul(W) + W.matmul(DfDx.transpose(1,2)) + 2 * _lambda * W
        C1_LHS_1 = _Bbot.transpose(1,2).matmul(C1_inner).matmul(_Bbot) # this has to be a negative defined matrix

        # C2
        C2_inners = []
        C2s = []
        for j in range(self.u_dim):
            C2_inner = weighted_gradients(W, B[:,:,j].unsqueeze(-1), x) - (DBDx[:,:,:,j].matmul(W) + W.matmul(DBDx[:,:,:,j].transpose(1,2)))
            C2 = _Bbot.transpose(1,2).matmul(C2_inner).matmul(_Bbot)
            C2_inners.append(C2_inner)
            C2s.append(C2)

        loss = 0
        epsilon = _lambda * 0.1
        loss += positive_defined_matrix(-Contraction- epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.type()))
        loss += positive_defined_matrix(-C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.type()))
        loss += positive_defined_matrix(w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type(x.type()) - W)
        loss += 1. * sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s])

        constraction_metric = ((torch.linalg.eigvalsh(Contraction,UPLO='U')>=0).sum(dim=1)==0).cpu().detach().numpy()
        C1_LHS_1_metric = ((torch.linalg.eigvalsh(C1_LHS_1,UPLO='U')>=0).sum(dim=1)==0).cpu().detach().numpy()
        C2_metric = sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s]).item()

        return loss, constraction_metric, C1_LHS_1_metric, C2_metric