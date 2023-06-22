""" iadb. added args, reduced number of checkpoints and images saved
if run as python `iadb.py celeba` should be identical to original

Added args, see 'run' function
"""
from typing import Union, Callable
from time import time
import os
import os.path as osp
import argparse
import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
from nndev import mean_center, mean_uncenter, blend_loss, make_noise, deblend, L2, L1, L2M, L2B, Perceptual


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
    parser.add_argument('-s', '--save_every', type=int, default=1000, help='save every [] iters')
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='checkpoint')
    parser.add_argument('-x', '--max_samples', type=int, default=8, help='max test samples')
    parser.add_argument('-o', '--objective', type=str, default='L2',
                        help='loss func: L1, L2, Perceptual')
    parser.add_argument('-m', '--center_mode', type=int, default=0,
                        help='0: x*2-1, 1: (x-imgnet.mean)/imgnet.std')
    parser.add_argument('-t', '--test', required=False, action='store_true')
    parser.set_defaults(test=False)
    return parser.parse_args()


def get_model(checkpoint: str, device: torch.device, lr_rate: float) -> tuple:
    """ original function + mod:
            added checkpoint arg, load checkpoint by name
            added optimizer & opt heckpoint
    """
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
    model = UNet2DModel(block_out_channels=block_out_channels, out_channels=3, in_channels=3,
                        up_block_types=up_block_types, down_block_types=down_block_types,
                        add_attention=True)
    if checkpoint:
        assert osp.isfile(checkpoint), f"checkpoint '{checkpoint}' not found"
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded model checkpoint: {checkpoint}")

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr_rate)
    if checkpoint:
        checkpoint = "_adam".join(osp.splitext(checkpoint))
        if osp.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            optimizer.load_state_dict(state_dict)
            print(f"Loaded optimizer checkpoint: {checkpoint}")

    return model, optimizer

# @torch.no_grad()
# def sample_iadb(model, x0, nb_step):
#     x_alpha = x0
#     for t in range(nb_step):
#         alpha_start = t/nb_step
#         alpha_end = (t+1)/nb_step
#         d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
#         x_alpha = x_alpha + (alpha_end-alpha_start)*d
#     return x_alpha


def run(data_folder: str = '.',
        results: str = '.',
        batch_size: int = 64,
        image_size: int = 64,
        epochs: int = 100,
        lr_rate: float = 1e-4,
        save_every: int = 1000,
        checkpoint: str = '',
        max_samples: int = 8,
        objective: str = 'L2',
        center_mode: int = 0,
        test: bool = False) -> None:
    """ main function, encapsulated.  Default is similar to original repo.

    added Args w comments
        data_folder (str) folder where celeba is or will be stored
        results     (str) folder where results are stored
        batch_size  (int [64]) tested  w larger batches. 4x batch size reduces training only 85% 
        image_size  (int [64]) variation untested... - may require different model
        epochs      (int [100]) the default 100 epochs over celebA may not suffice
        lr_rate     (float [1e-4])
        save_every  (int [1000]) default was 200
        checkpoint  (str [''])  load to continue training -*deso not store optimizer momentum 
        max_samples (int [16]) inference samples can consume significant portion of training time
        objective   (str [L2]) L1 | Perceptual
        test        (bool [False]) if True, returns a single inference instance
            test without training, args: checkpoint=, test=,

    Differences with original.
        inference outputs images used to train rnd state + sampled. To explore correlations
        store simple traing log iadb.csv logging epoch, instance, loss, time.

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(image_size),transforms.CenterCrop(image_size),
                                    transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
    train_dataset = torchvision.datasets.CelebA(root=data_folder, split='train',
                                            download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0, drop_last=True)
    nb_iter = 0
    nb_epoch = 0
    if checkpoint:
        _nb_iter = osp.splitext(osp.basename(checkpoint))[0].split("_")[-1]
        if _nb_iter.isnumeric():
            nb_iter = int(_nb_iter)
            nb_epoch = 1 + int(nb_iter/len(dataloader))
        if not osp.isfile(osp.abspath(osp.expanduser(checkpoint))):
            checkpoint = osp.join(results, checkpoint)
    model, optimizer = get_model(checkpoint, device=device, lr_rate=lr_rate)


    if test:
        print(f"Test on checkpoint {osp.basename(checkpoint)}")
    else:
        print(f'Start training on iter {nb_iter}')

    loss_fun = get_loss_fun(objective, device)

    start = time()
    for current_epoch in range(epochs):
        for i, data in enumerate(dataloader):
            x1 = mean_center(data[0], center_mode).to(device=device)
            x0 = make_noise(x1)
            loss = blend_loss(model, x1, noise=x0, loss_fun=loss_fun, mask=None, divisor=None)

            # log training loss and time
            write_log(current_epoch,  nb_epoch, i, loss, start, results, nb_iter)

            # train step
            if not test:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                nb_iter += 1

            # save checkpoint and inference sample
            if (nb_iter % save_every == 0) or test or nb_iter==100:
                with torch.no_grad():
                    print(f'Save export {nb_iter}')

                    sample = mean_uncenter(deblend(model, x0[:max_samples], 128).cpu(), center_mode)
                    sample = torch.cat((data[0][:max_samples], sample))
                    name = f'export__{nb_iter:08d}.png'
                    if checkpoint and test:
                        name = f'export_{osp.basename(checkpoint)}.png'
                    torchvision.utils.save_image(sample, osp.join(results, name))
                    if test:
                        return
                    # name = osp.join(results, f'celeba_{batch_size}_{nb_iter:08d}.ckpt')
                    name = osp.join(results, 'celeba_iadb.ckpt')
                    torch.save(model.state_dict(), name)
                    name = osp.join(results, 'celeba_iadb_adam.ckpt')
                    torch.save(optimizer.state_dict(), name)


def get_loss_fun(objective: str, device: Union[torch.device, str]) -> Callable:
    if objective == 'Perceptual':
        return Perceptual(device=device).lose
    if objective == "L1":
        return L1
    return L2


def write_log(current_epoch,  nb_epoch, i, loss, start, results, nb_iter):
    """log training loss and time"""
    _state = f"{current_epoch+nb_epoch},{i},{loss.item():.05f},{time()-start:.1f}"
    with open(osp.join(results, 'iadb.csv'), 'a', encoding='utf8') as _fi:
        if nb_iter == 0:
            _fi.write("epoch,iter,loss,time\n")
        _fi.write(_state + "\n")
        print(_state.replace(",", "\t"))


if __name__ == "__main__":
    args = parse_args()
    DATASET = osp.abspath(osp.expanduser(args.data_folder))
    RESULTS = osp.abspath(osp.expanduser(osp.join(args.results, args.name)))
    if not args.checkpoint:
        assert not osp.isdir(RESULTS), f"experiment {RESULTS} exists, delete or rename"
        os.makedirs(RESULTS, exist_ok=True)

    run(DATASET, RESULTS, args.batch_size, args.image_size, args.epochs, args.lr_rate,
        args.save_every, args.checkpoint, args.max_samples, args.objective,
        args.center_mode, args.test)
