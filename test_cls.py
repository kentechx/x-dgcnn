import click
import torch.nn.functional as F
import wandb
from x_dgcnn import XEdgeConv, XSpatialTransformNet, XDGCNN_Cls, DGCNN_Cls, DGCNN_Seg

from thop import profile, clever_format
from tqdm import tqdm

import torch


def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


@click.command()
@click.option('--model', type=str, default='xdgcnn')
@click.option('--batch_size', type=int, default=32)
@click.option('--n_points', type=int, default=1024)
@click.option('--k', type=int, default=20)
@click.option('--xdgcnn_k', default=4)
@click.option('--sampling_ratio', type=tuple, default=(4, 16, 64, 256))
@click.option('--device', default='cuda')
@click.option('--offline', is_flag=True, default=False)
def run(model, batch_size, n_points, k, xdgcnn_k, sampling_ratio, device, offline):
    config = {'model': model,
              'batch_size': batch_size,
              'n_points': n_points,
              'k': k,
              'xdgcnn_k': xdgcnn_k,
              'sampling_ratio': sampling_ratio,
              'device': device}

    if model == 'dgcnn':
        model = DGCNN_Cls(k=k, in_dim=3, out_dim=40).to(device)
    elif model == 'xdgcnn':
        model = XDGCNN_Cls(in_dim=3, out_dim=40,
                           base_points=n_points, sampling_ratio=sampling_ratio, k=xdgcnn_k).to(device)
    else:
        raise NotImplementedError

    wandb.init(project='xdgcnn_random_test', name=config['model'], config=config,
               mode='online' if not offline else 'disabled'
               )

    seed_everything(0)

    x = torch.randn(batch_size, 3, 1024, device=device)
    xyz = x[:, :3, :].clone()
    y = torch.randint(0, 40, (batch_size,), device=device)

    # print macs and params
    macs, params = profile(model, inputs=(x, xyz))
    print(clever_format((macs, params), "%.3f"))

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    wandb.watch(model, creterion=F.cross_entropy, log_freq=100, log='gradients', log_graph=False)
    wandb.log({'macs': macs, 'params': params})

    bar = tqdm(range(2000))
    for i in bar:
        optim.zero_grad()
        out = model(x, xyz)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optim.step()

        wandb.log({'loss': loss})
        bar.set_description(f'loss: {loss.item():.8f}')

    wandb.finish()


if __name__ == '__main__':
    run()
