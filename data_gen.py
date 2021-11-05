"""
Create synthetic 1D data

target is a color curve on the plane: f(x, y) = 0
view from the plane: (x, y, theta)
"""
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import math

def curve_fn(x, y):
    """
    Implicit function f(x, y). If f(x, y) == 0 then it is on the curve. 
    """
    return x**2 + y**2 - 1 

def curve_color(x, y):
    """
    The color (RGB) on the curve at position (x, y) 
    """
    # assume return a color value between (0, 1)
    mapper = cm.ScalarMappable(cmap='rainbow')
    return mapper.to_rgba(x)[:3]


def get_dirs(fov):
    """[summary]

    Returns:
        [type]: [description]
    """    
    angle_min, angle_max = fov
    angle = torch.linspace(angle_min, angle_max, angle_max - angle_min + 1)
    return torch.vstack([torch.cos(angle), torch.sin(angle)])


def get_rays(fov, c2w):
    """
    Start from camera center, sample points from fov along certain depth distance,
    then transform the points to the world coordinates

    For the camera coordinates, y is pointing to the world center and x forms a 
    right-hand coordinates with y facing outward

    Args:
        fov (N, 2): [description]
        c2w (N, 3, 3): [description]
    
    Returns:
        rays_d (N, 2) direction rays
        rays_o (N, 2) origin of rays
    """
    
    # sample distance along each rays_d
    dirs = get_dirs(fov)

    R, t = c2w[:2, :2], c2w[:2, -1]
    rays_d = R @ dirs # (2, N)
    rays_o = t.expand(rays_d.shape)
    return rays_d.t(), rays_o.t()


def get_trans_c2w(r, theta):
    tx = r * torch.cos(theta)
    ty = r * torch.sin(theta)
    R = torch.zeros([r.shape[:-1]] + [2, 2])
    new_theta = theta + math.pi
    R[..., 0, 0] = torch.cos(new_theta)
    R[..., 0, 1] = -torch.sin(new_theta)
    R[..., 1, 0] = torch.cos(new_theta)
    R[..., 1, 1] = torch.sin(new_theta)
    c2w = torch.zeros([r.shape[:-1]] + [2, 3])
    c2w[..., :2, :2] = R
    c2w[..., 0, 2:] = tx
    c2w[..., 1, 2:] = ty
    return c2w
 
def sample_camera_pose(n_samples, near=1.5, far=5):
    """
    Sample a few c2w to generate view data
    """
    # sample points in polar coordinates
    
    r_rand  = torch.rand(n_samples)
    r = (1 - r_rand) * near + r_rand * far
    
    theta = torch.linspace(0, 2 * math.pi, n_samples)

    c2w = get_trans_c2w(r, theta)
    return c2w
    





if __name__ == "__main__":
    pass
