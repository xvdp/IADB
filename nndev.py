
"""@xvdp
iadb training
is L2 Loss best?
"""
from typing import Optional, Union, Callable
from functools import partial
import torch
from torch import nn, Tensor
from torchvision import models, transforms

# pylint: disable=no-member
# pylint: disable=invalid-name

def L2(pred: Tensor,
       data: Tensor,
       mask: Optional[Tensor] = None,
       divisor: Union[str, int, float, None] = None) -> Tensor:
    """Masked L2 loss
    Args
        mask    same size as 
        divisor (int [1]) 1: L2, len(pred): L2 per item, pred.numel(): MSE
    """
    if divisor is None:
        divisor = 1
    elif isinstance(divisor, str):
        if divisor.lower()[0] == 'm': # mean
            divisor = pred.numel()
        elif divisor.lower()[0] == 'b': # batch mean
            divisor = len(pred)
        else:
            raise NotImplementedError(f"\divisor: int > 0, 'mean' or 'batch', got {divisor}")
    return torch.sum(submask((pred - data)**2, mask))/divisor

L2M = partial(L2, divisor='mean') # mean square loss
L2B = partial(L2, divisor='batch')

def L1(pred: Tensor, data: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    return torch.sum(submask(pred - data, mask))


class Perceptual:
    """ mini masked perceptual loss over vgg16 layer 9
    """
    def __init__(self, device=None):
        print ("Loading perceptual loss features model")
        self.net = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9].to(device=device)
        self.net.eval()
        self.net.requires_grad_(False)
    def lose(self,
             x: Tensor,
             y: Tensor,
             mask: Optional[Tensor] = None,
             divisor: Optional[int] = None):
        """
        Args:
            x, y    Tensors fof same type
            mask    Tensor premultiply to ignore areas
        """
        assert x.shape == y.shape and x.device == y.device and x.dtype == y.dtype, \
            f"expected similar tensors got {x.shape}:{y.shape}, {x.dtype}:{y.dtype},\
                {x.device}:{y.dtype}"
        _n = len(x)
        out = self.net(torch.cat((submask(x, mask), submask(y, mask))))
        divisor = divisor or out[:_n].numel()
        return L2(out[:_n], out[_n:], divisor=divisor)


def submask(data: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """mask data"""
    if mask is not None:
        assert data.shape == mask.shape, f"incorrect mask shape {mask.shape}, {data.shape}"
        return data * mask.to(device=data.device, dtype=data.dtype)
    return data


def mean_center(data, mode=0):
    """
    mode: 0 - iadb:         data * 2 -1
          1 - imagenet      (data - imagenet.mean) /imagenet.std
    """
    if mode == 0:
        return data * 2 - 1
    elif mode == 1: # Imagenet
        assert data.ndim == 4 and data.shape[1] == 3
        mean=torch.Tensor([0.485, 0.456, 0.406], device=data.device).view(1,-1,1,1)
        std=torch.Tensor([0.229, 0.224, 0.225], device=data.device).view(1,-1,1,1)
        return (data - mean) / std
    raise NotImplementedError(f'mode in 1 or 2, got {mode}')


def mean_uncenter(data, mode=0):
    if mode == 0:
        return (data + 1) /2
    elif mode == 1: # Imagenet
        assert data.ndim == 4 and data.shape[1] == 3
        mean=torch.Tensor([0.485, 0.456, 0.406], device=data.device).view(1,-1,1,1)
        std=torch.Tensor([0.229, 0.224, 0.225], device=data.device).view(1,-1,1,1)
        return data * std + mean
    raise NotImplementedError(f'mode in 1 or 2, got {mode}')


def make_noise(data: Tensor, noise: Optional[Tensor] = None) -> Tensor:
    """ make gaussian noise optionally
    """
    if noise is None:
        return torch.randn_like(data)
    assert noise.shape == data.shape, \
        f"noise {noise.shape} and data {data.shape} must have same shape"
    return noise.to(dtype=data.dtype, device=data.device)


def blend_loss(model: nn.Module,
          data: Tensor,
          noise: Optional[Tensor] = None,
          loss_fun: Callable = L2,
          mask: Optional[Tensor] = None,
          divisor: Optional[int] = None) -> Tensor:
    """ training loss
    Args
        model       (nn.Module) denoising net
        data        (tensor) training input: ndim and device must match model
            must be mean centered
        noise       (Tensor [None]) noise distribution default: normal
        mask        (Tensor [None])
        divisor (int [1])
            1:              sum, L2
            len(pred):      sum/batch_size  batch mean
            pred.numel():   sum/numel, MSE  item mean 
    """
    noise = make_noise(data, noise)
    alpha = torch.rand(len(data), dtype=data.dtype, device=data.device)
    noised_data = torch.lerp(noise, data, alpha.view(-1, *[1]*(data.ndim-1)))
    logits = model(noised_data, alpha)['sample']
    return loss_fun(logits, data - noise, mask, divisor)


# from torchvision import models, transforms
# mod = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9]

@torch.no_grad()
def deblend(model: nn.Module, io: Tensor, nb_step: int = 128) -> Tensor:
    """ in place sample
    Args
        model   (nn.Module) trained denoising net
        io      (Tensor) trained input distribution sample, e.g. gaussian
        nb_step (int) number of denoising steps
    """
    for alpha in torch.linspace(0, 1, nb_step + 1, device=io.device, dtype=io.dtype)[:-1]:
        io = io + model(io, alpha)['sample'] / nb_step
    return io
