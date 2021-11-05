"""
Visualize data and results
"""
import pdb
import torch
import math
import numpy as np
from model import NeRF
from run_nerf import render_rays
from matplotlib import cm, colors
import matplotlib.pyplot as plt


def render_path_circle(model, cuda=True):
    """
    Sample points along the circle and render what is the view
    """
    r = 3
    theta = torch.linspace(0, 2 * math.pi, 360)
    x, y = r * torch.cos(theta), r * torch.sin(theta)
    rays_o = torch.stack([x, y]).t()
    rays_d = -theta
    view_dirs = rays_d
    if cuda:
        rays_o, rays_d, view_dirs = rays_o.cuda(), rays_d.cuda(), view_dirs.cuda()
    res = render_rays(rays_o, rays_d, view_dirs, model.forward)
    return x, y, res

def vis_res(x, y, pred):
    # norm = colors.Normalize(vmin=0, vmax=1)
    # mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
    color = pred #mapper.to_rgba(pred)[..., :3]
    plt.scatter(x, y, c=color)
    plt.axis('equal')
    plt.savefig('res.png', dpi=500)


def vis_data(x, y, theta, c):
    """
    Visualize the data before training
    """
    plt.scatter(x, y, c=c)
    plt.quiver(x, y, np.cos(theta), np.sin(theta))
    plt.savefig('data.png', dpi=500)



if __name__ == "__main__":
    model = NeRF.load_from_checkpoint('lightning_logs/version_34_10/checkpoints/epoch=401-step=401.ckpt')
    model.eval()
    model.cuda()
    with torch.no_grad():
        x, y, res = render_path_circle(model)
    res = res.cpu()
    # import pdb; pdb.set_trace()
    vis_res(x, y, res)
    
    # from dataloader import OneDCurveData
    # dataset = OneDCurveData()
    # x, y, theta, c = dataset.x, dataset.y, dataset.view, dataset.color

    # vis_data(x, y, theta, c)







