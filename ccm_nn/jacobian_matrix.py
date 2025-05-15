'''
@Filename       : jacobian_matrix.py
@Description    : 
@Create_Time    : 2025/02/17 10:55:11
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def jacobian(y: torch.Tensor, x: torch.Tensor, need_higher_grad=True) -> torch.Tensor:
    """compute the jacobian matrix

    Args:
        y (torch.Tensor): output tensor (n1 x n2)
        x (torch.Tensor): input tensor (m1 x m2)
        need_higher_grad (bool, optional): Defaults to True.

    Returns:
        torch.Tensor: jacobian matrix (n1 x n2 x m1 x m2)
    """
    y = y.requires_grad_()
    (Jac,) = torch.autograd.grad(
        outputs=(y.flatten(),),
        inputs=(x,),
        grad_outputs=(torch.eye(torch.numel(y)).to(device),),
        create_graph=need_higher_grad,
        allow_unused=True,
        is_grads_batched=True
    )
    if Jac is None:
        Jac = torch.zeros(size=(y.shape + x.shape))
    else:
        Jac.reshape(shape=(y.shape + x.shape))
    return Jac.squeeze(-1)

def batched_jacobian(batched_y:torch.Tensor,batched_x:torch.Tensor,need_higher_grad = True) -> torch.Tensor:
    """compute the jacobian matrix of batched matrix 

    Args:
        batched_y (torch.Tensor): batched matrix_y (bs x n1 x n2)
        batched_x (torch.Tensor): batched matrix_x (bs x m1 x 1)
        need_higher_grad (bool, optional): Defaults to True.

    Returns:
        torch.Tensor: jacobian matrix (bs x n1 x n2 x m1)
    """
    sumed_y = batched_y.sum(dim = 0) 
    J = jacobian(sumed_y,batched_x,need_higher_grad) # shape of J:(n1*n2 x bs x m1)

    m1 = batched_x.shape[1]
    
    dims = list(range(J.dim()))
    dims[0],dims[1] = dims[1],dims[0]
    J = J.permute(dims = dims) # shape of J:(bs x n1*n2 x m1)

    if (batched_y.dim() == 3) & (batched_y.shape[2] != 1):
        (bs,n1,n2) = batched_y.shape
        J = J.reshape(bs,n1,n2,m1)
    if ((batched_y.dim() == 3) & (batched_y.shape[2] == 1)):
        (bs,n1,n2) = batched_y.shape
        J = J.reshape(bs,n1,m1)
    if (batched_y.dim() == 2):
        (bs,n1) = batched_y.shape
        J = J.reshape(bs,n1,m1)
    return J

def weighted_gradients(W, v, x, detach=False):
    # v, x: bs x n x 1
    # DWDx: bs x n x n x n
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (batched_jacobian(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (batched_jacobian(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

def Bbot_func(x, state_dim, u_dim): # columns of Bbot forms a basis of the null space of B^T
        bs = x.shape[0]
        Bbot = torch.cat((torch.eye(state_dim-u_dim, state_dim-u_dim),
            torch.zeros(u_dim, state_dim-u_dim)), dim=0).type(x.type())
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)