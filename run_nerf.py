import torch
import torch.nn.functional as F
cuda = True
def linear_sample(near, far, num_samples):
    t_vals = torch.linspace(0, 1, steps=num_samples)
    return near * (1 - t_vals) + far * t_vals

def stratified_sample(sample):
    """In each interval sample[i-1], sample[i] of sample, randomly select a number within the interval as the new sample

    Args:
        sample (..., N): original sample

    Returns:
        new_sample (..., N): new sample after perturbing
    """
    left_bin_edge = sample[..., :-1]   
    right_bin_edge = sample[..., 1:]
    mids = 0.5 * (left_bin_edge + right_bin_edge)
    lower = torch.cat((left_bin_edge[..., :1], mids), -1)
    upper = torch.cat((mids, right_bin_edge[-1:]), -1)
    t_rand = torch.rand_like(sample)
    return lower + (upper - lower) * t_rand

def sigma2alpha(sigma, dist):
    return 1. - torch.exp(-F.relu(sigma) * dist)

def raw2outputs(raw, z_vals):
    dists = z_vals[...,1:] - z_vals[...,:-1]
    if cuda:
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).cuda()], -1)
    else:
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)
    
    rgb = torch.sigmoid(raw[...,:3])
    alpha = sigma2alpha(raw[...,3], dists)
    
    # import pdb; pdb.set_trace()
    if cuda:
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[..., :-1]
    else:
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[..., :-1]
    # import pdb; pdb.set_trace()
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    return rgb_map    

def render_rays(rays_o, rays_d, view_dirs, network_fn):
    near = 0
    far = 4
    n_samples = 128
    if cuda:
        z_vals = linear_sample(near, far, n_samples).cuda()
    else:
        z_vals = linear_sample(near, far, n_samples)
    
    # import pdb; pdb.set_trace()
    z_vals = stratified_sample(z_vals)
    # pts = rays_o[...,None,:] + torch.stack([torch.cos(rays_d)[...,None] * z_vals[None, ...], torch.sin(rays_d)[...,None] * z_vals[None, ...]], -1)
    # raw = network_fn(torch.cat([pts, view_dirs[...,None, None].expand((-1, n_samples, 1))],-1))
    
    # import pdb; pdb.set_trace()
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[None, ..., None]#torch.stack([torch.cos(rays_d)[...,None] * z_vals[None, ...], torch.sin(rays_d)[...,None] * z_vals[None, ...]], -1)
    raw = network_fn(torch.cat([pts, view_dirs[...,None, :].expand((-1, n_samples, 2))],-1))
    
    return raw2outputs(raw, z_vals)

def raw2outputs_step(raw, z_vals):
    dists = z_vals[...,1:] - z_vals[...,:-1]
    if cuda:
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).cuda()], -1)
    else:
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)
    
    rgb = torch.sigmoid(raw[...,:3])
    alpha = sigma2alpha(raw[...,3], dists)
    
    # import pdb; pdb.set_trace()
    if cuda:
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[..., :-1]
    else:
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[..., :-1]
    # import pdb; pdb.set_trace()
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    return rgb_map, weights, rgb   

def render_rays_step(rays_o, rays_d, view_dirs, network_fn):
    near = 0
    far = 4
    n_samples = 64
    if cuda:
        z_vals = linear_sample(near, far, n_samples).cuda()
    else:
        z_vals = linear_sample(near, far, n_samples)
    # import pdb; pdb.set_trace()

    pts = rays_o[...,None,:] + torch.stack([torch.cos(rays_d)[...,None] * z_vals[None, ...], torch.sin(rays_d)[...,None] * z_vals[None, ...]], -1)
    raw = network_fn(torch.cat([pts, view_dirs[...,None, None].expand((-1, n_samples, 1))],-1))
    return raw, z_vals


    
    