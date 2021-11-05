from torch.utils.data import Dataset
import torch
import math
from matplotlib import cm

def curve_fn(x, y):
    """
    Implicit function f(x, y). If f(x, y) == 0 then it is on the curve. 
    """
    return x**2 + y**2 - 1 

def curve_color(x, y):
    """
    The color (RGB) on the curve at position (x, y) 
    """
    assert torch.all(curve_fn(x, y) < 1e-5)
    # normalize 
    val = torch.atan2(y, x)
    mapper = cm.ScalarMappable(cmap='rainbow')
    return mapper.to_rgba(val)[..., :3]


class OneDCurveData(Dataset):
    def __init__(self) -> None:
        super().__init__()
        near = 1.5 
        far = 4
        self.n_samples = n_samples = 20
        r_rand  = torch.rand(n_samples)
        r = (1 - r_rand) * near + r_rand * far
        theta = torch.rand(n_samples) * 2 * math.pi
        self.x = r * torch.cos(theta)
        self.y = r * torch.sin(theta)
        self.view = theta + math.pi 
        self.color = torch.from_numpy(curve_color(self.x/r, self.y/r))
        # self.color[torch.abs(curve_fn(self.x, self.y)) > 1e-1] = 0
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.stack([self.x[idx], self.y[idx], self.view[idx]]), self.color[idx]
