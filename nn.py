"""@xvdp
iadb training and sampling functions
model and data ndim independent
"""
from typing import Optional
import torch
from torch import nn, Tensor

# pylint: disable=no-member
# pylint: disable=invalid-name
@torch.no_grad()
def deblend(model: nn.Module, io: Tensor, nb_step: int = 128) -> Tensor:
    """ in place sample
    Args
        model   (nn.Module) trained denoising net
        io      (Tensor) trained input distribution sample, e.g. gaussian
        nb_step (int) number of denoising steps
    :.io and model need to be on same device
    """
    for alpha in torch.linspace(0, 1, nb_step + 1, device=io.device, dtype=io.dtype)[:-1]:
        io = io + model(io, alpha)['sample'] / nb_step
    return io


def blend_loss(model: nn.Module, data: Tensor, noise: Optional[Tensor] = None) -> Tensor:
    """ training loss
    Args
        model   (nn.Module) denoising net
        data    (tensor) training input
        noise   (Tensor [None]-> normal) noise distribution to train the model on
    data and model need to be on same device
    data.ndim must match model required ndim
    """
    data = (data * 2) - 1 # center to zero
    noise = make_noise(data, noise)
    alpha = torch.rand(len(data), dtype=data.dtype, device=data.device)
    noised_data = torch.lerp(noise, data, alpha.visew(-1, *[1]*(data.ndim - 1)))
    out = model(noised_data, alpha)['sample']
    return torch.sum((out - (data - noise))**2)


def make_noise(data: Tensor, noise: Optional[Tensor] = None) -> Tensor:
    """ make gaussian noise optionally
    """
    if noise is None:
        return torch.randn_like(data)
    assert noise.shape == data.shape, \
        f"noise {noise.shape} and data {data.shape} must have same shape"
    return noise.to(dtype=data.dtype, device=data.device)
