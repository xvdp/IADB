"""@xvdp
iadb training and sampling functions
"""
import torch
from torch import nn, Tensor
from torch.optim import Optimizer

# pylint: disable=no-member
@torch.no_grad()
def deblend(io: Tensor, model: nn.Module, nb_step: int = 128) -> Tensor:
    """ in place sample
    Args
        model   (nn.Module) trained denoising net
        io      (Tensor) trained input distribution sample, e.g. gaussian
        nb_step (int) number of denoising steps
    """
    step = 1/nb_step
    alphas = torch.linspace(0, 1, nb_step + 1, device=io.device)[:-1]
    for _, alpha in enumerate(alphas):
        io = io + model(io, alpha)['sample'] * step
    return io


def blend_loss(data: Tensor, model: nn.Module) -> Tensor:
    """ training loss
    Args
        data    (tensor) training input
        model   (nn.Module) denoising net
     """
    data = (data * 2) - 1 # center to zero
    noise = torch.randn_like(data)
    alpha = torch.rand(len(data), device=data.device)
    noised_data = torch.lerp(noise, data, alpha.view(-1,1,1,1))
    out = model(noised_data, alpha)['sample']
    return torch.sum((out - (data - noise))**2)

def train_step(optimizer: Optimizer, loss: Tensor, iter: int = 0) -> int:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return iter + 1