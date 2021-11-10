"""
Create synthetic 1D data

target is a color curve on the plane: f(x, y) = 0
view from the plane: (x, y, theta)
"""
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import math

# ref: https://mathworld.wolfram.com/Circle-LineIntersection.html
def circle_line_intersect(p1, p2, center, r):
    """
    Find the intersection of circle defined by (center, r) and the line defined by two points on it (p1, p2)
    p1: (N, 2)
    p2: (N, 2)
    center (2,)
    r (1,)
    """
    # shift the coordinate origin
    p1_new = p1 - center
    p2_new = p2 - center

    dxy = p2_new - p1_new
    D = p1_new[:, 0] * p2_new[:, 1] - p1_new[:, 1] * p2_new[:, 0]
    dr_2 = (dxy**2).sum(-1)
    
    delta = r**2 * dr_2 - D**2 
    has_sol = delta > 0
    D_good, dxy_good, dr_2_good, delta_good = list(map(lambda x: x[has_sol], [D, dxy, dr_2, delta]))
    x1 = (D_good * dxy_good[:, 1] + torch.sign(dxy_good[:, 1]+1e-5) * dxy_good[:, 0] * torch.sqrt(delta_good)) / dr_2_good
    y1 = (-D_good * dxy_good[:, 0] + torch.abs(dxy_good[:, 1]) * torch.sqrt(delta_good)) / dr_2_good
    x2 = (D_good * dxy_good[:, 1] - torch.sign(dxy_good[:, 1]+1e-5) * dxy_good[:, 0] * torch.sqrt(delta_good)) / dr_2_good
    y2 = (-D_good * dxy_good[:, 0] - torch.abs(dxy_good[:, 1]) * torch.sqrt(delta_good)) / dr_2_good
    # assume p1 is the start point
    y_ori = p1_new[:, 1][has_sol]
    x = torch.where(y_ori > 0, x1, x2)
    y = torch.where(y_ori > 0, y1, y2)
    return x, y, has_sol



def curve_fn(x, y):
    """
    Implicit function f(x, y). If f(x, y) == 0 then it is on the curve. 
    """
    return x**2 + y**2 - 1 

def curve_color(x, y):
    """
    The color (RGB) on the curve at position (x, y) 
    """
    assert torch.all(curve_fn(x, y) < 1e-2)
    # normalize 
    val = torch.atan2(y, x)
    mapper = cm.ScalarMappable(cmap='rainbow')
    return mapper.to_rgba(val)[..., :3]


def get_dirs(fov, step_deg=5):
    """given fov as min and max angle, return intermediate unit direction
    with step size = one degree

    Returns:
        (2, N): N unit direction 
    """    
    angle_min, angle_max = fov
    steps = int((angle_max - angle_min)/math.pi * 180)//step_deg + 1
    angle = torch.linspace(angle_min, angle_max, steps)
    return torch.vstack([torch.cos(angle), torch.sin(angle)])


def get_rays(fov, c2w):
    """
    Start from camera center, sample points from fov along certain depth distance,
    then transform the points to the world coordinates

    For the camera coordinates, y is pointing to the world center and x forms a 
    right-hand coordinates with y facing outward

    Args:
        fov (2): [description]
        c2w (N, 3, 3): [description]
    
    Returns:
        rays_d (N, 2) direction rays
        rays_o (N, 2) origin of rays
    """
    
    # sample distance along each rays_d
    dirs = get_dirs(fov, step_deg=1)
    # sample directions, takes only a few in fov
    # import pdb; pdb.set_trace()
    # perm = torch.randperm(dirs.shape[1])
    # idx = perm[:3]
    # dirs = dirs[:, idx]

    R, t = c2w[..., :2, :2], c2w[..., :2, -1]
    rays_d = (R @ dirs)
    rays_o = t[...,None].expand(rays_d.shape)

    rays_d, rays_o = rays_d.transpose(1, 2).reshape(-1, 2), rays_o.transpose(1, 2).reshape(-1, 2)
    return rays_d, rays_o


def get_trans_c2w(r, theta):
    """
    

    Args:
        r (N): sample distance
        theta (N): sample angle

    Returns:
        c2w (N, 2, 3): 
    """
    tx = r * torch.cos(theta)
    ty = r * torch.sin(theta)
    c2w = torch.zeros(r.shape + (2, 3))

    new_theta = theta + math.pi
    c2w[..., 0, 0] = torch.cos(new_theta)
    c2w[..., 0, 1] = -torch.sin(new_theta)
    c2w[..., 1, 0] = torch.sin(new_theta)
    c2w[..., 1, 1] = torch.cos(new_theta)
    
    c2w[..., 0, 2:] = tx.reshape(-1, 1)
    c2w[..., 1, 2:] = ty.reshape(-1, 1)
    return c2w
 
def sample_camera_pose(n_samples, near=1.5, far=5):
    """
    Sample a few c2w to generate view data
    """
    # sample points in polar coordinates
    
    r_rand  = torch.rand(n_samples)
    r = (1 - r_rand) * near + r_rand * far # uniformly sample from [near, far]
    
    theta = torch.rand(n_samples) * 2 * math.pi  # uniformly sample from [0, 2*pi]

    c2w = get_trans_c2w(r, theta)
    return c2w


def sample_camera_pose_more_depth(n_samples, near=1.5, far=5):
    """
    Sample a few c2w to generate view data
    """
    # sample points in polar coordinates
    
    r_rand  = torch.rand(n_samples)
    mid = .5 * (near + far)
    r = (1 - r_rand) * near + r_rand * mid # uniformly sample from [near, far]
    
    theta = torch.rand(n_samples) * 2 * math.pi  # uniformly sample from [0, 2*pi]

    c2w1 = get_trans_c2w(r, theta)

    r = (1 - r_rand) * mid + r_rand * far # uniformly sample from [near, far]
    c2w2 = get_trans_c2w(r, theta)
    
    c2w = torch.cat([c2w1, c2w2])

    return c2w


if __name__ == "__main__":
    n_samples = 10
    c2w = sample_camera_pose(n_samples)
    fov = torch.tensor([-math.pi/6, math.pi/6])
    rays_d, rays_o = get_rays(fov, c2w)
    x, y = rays_o[:, 0], rays_o[:, 1]
    u, v = rays_d[:, 0], rays_d[:, 1]
    plt.scatter(x, y)
    plt.quiver(x, y, u, v)
