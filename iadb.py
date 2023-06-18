""" iadb. added args, reduced number of checkpoints and images saved
if run as python `iadb.py celeba` should be identical to original
"""
from typing import Union
from time import time
import os
import os.path as osp
import argparse
import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam


# pylint: disable=no-member
def parse_args():
    """xvdp added parser"""
    parser = argparse.ArgumentParser(description='IADB args')
    parser.add_argument('data_folder', type=str, default=".", help='Path to the data folder')
    parser.add_argument('-r', '--results', type=str, default='./results',
                        help='Path to the results folder')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-i', '--image_size', type=int, nargs='+', default=64,
                        help='Image size (single integer or tuple)')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-l', '--lr_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-n', '--name', type=str, default="iadb", help='experiment name')
    parser.add_argument('-s', '--save_every', type=int, default=200, help='save every [] iters')
    args = parser.parse_args()
    return args

def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3, in_channels=3,
                       up_block_types=up_block_types, down_block_types=down_block_types,
                       add_attention=True)

@torch.no_grad()
def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)
        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

def train(data_folder: str = '.',
          results: str = '.',
          batch_size: int = 64,
          image_size: Union[int, tuple] = 64,
          epochs: int = 100,
          lr_rate: float = 1e-4,
          save_every: int = 1000) -> None:
    """ xvdp main f8ct90j
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(image_size),transforms.CenterCrop(image_size),
                                    transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
    train_dataset = torchvision.datasets.CelebA(root=data_folder, split='train',
                                            download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0, drop_last=True)
    model = get_model()
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr_rate)
    nb_iter = 0
    print('Start training')
    start = time()
    for current_epoch in range(epochs):
        for i, data in enumerate(dataloader):
            x1 = (data[0].to(device)*2)-1
            x0 = torch.randn_like(x1)
            bs = x0.shape[0]

            alpha = torch.rand(bs, device=device)
            x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0

            d = model(x_alpha, alpha)['sample']
            loss = torch.sum((d - (x1-x0))**2)

            #xvdp added log
            _state = f"{current_epoch},{i},{loss.item():.05f},{time()-start:.1f}"
            with open(osp.join(results, 'iadb.csv'), 'a', encoding='utf8') as _fi:
                if nb_iter == 0:
                    _fi.write("epoch,iter,loss,time\n")
                _fi.write(_state + "\n")
                print(_state.replace(",", "\t"))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nb_iter += 1

            if nb_iter % save_every == 0:
                with torch.no_grad():
                    print(f'Save export {nb_iter}')
                    sample = (sample_iadb(model, x0, nb_step=128) * 0.5) + 0.5
                    torchvision.utils.save_image(sample,
                                                 osp.join(results,
                                                          f'export_{str(nb_iter).zfill(8)}.png'))
                    torch.save(model.state_dict(),
                               osp.join(results, f'celeba_{current_epoch:02d}_{i:05d}.ckpt'))

if __name__ == "__main__":
    args = parse_args()
    DATASET = osp.abspath(osp.expanduser(args.data_folder))
    RESULTS = osp.abspath(osp.expanduser(osp.join(args.results, args.name)))
    assert not osp.isdir(RESULTS), f"experiment {RESULTS} exists, delete or rename"
    os.makedirs(RESULTS, exist_ok=True)

    BSIZE = args.batch_size
    ISIZE = args.image_size
    EPOCHS = args.epochs
    LR = args.lr_rate
    SAVE = args.save_every

    train(DATASET, RESULTS, BSIZE, ISIZE, EPOCHS, LR, SAVE)
