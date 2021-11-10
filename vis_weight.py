from run_nerf import *
import math
import numpy as np
import matplotlib.pyplot as plt

# sample 10 points, record their rays_o, rays_d, and predicted color and weights
r = 4
n_samples = 36
theta = torch.linspace(0, 2 * math.pi, n_samples)
x, y = r * torch.cos(theta), r * torch.sin(theta)
rays_o = torch.stack([x, y]).t()
rays_d = (theta + math.pi + math.pi) % (2 * math.pi) - math.pi
view_dirs = rays_d
rays_o, rays_d, view_dirs = rays_o.cuda(), rays_d.cuda(), view_dirs.cuda()
with torch.no_grad():
    raw, z_vals = render_rays_step(rays_o, rays_d, view_dirs, model.forward)
    y_hat, weight, rgb = raw2outputs_step(raw, z_vals)
pts = rays_o[...,None,:] + torch.stack([torch.cos(rays_d)[...,None] * z_vals[None, ...], torch.sin(rays_d)[...,None] * z_vals[None, ...]], -1)
pts = pts.cpu().detach()
theta = np.linspace(0, math.pi * 2, 360)
x_c, y_c =  np.cos(theta), np.sin(theta)
for i in range(n_samples):
    plt.scatter(pts[i, :, 0], pts[i, :, 1], c=weight[i].cpu())
plt.scatter(x_c, y_c, c='r', s=1)
plt.axis('equal')
    