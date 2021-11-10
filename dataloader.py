from torch.utils.data import Dataset
import torch
import math
from matplotlib import cm
from data_gen import *


# class OneDCurveNewData(Dataset):
#     def __init__(self, n_samples=1000) -> None:
#         super().__init__()
#         self.n_samples = n_samples
#         c2w = sample_camera_pose_more_depth(n_samples, near=1.5, far=5)
#         fov = torch.tensor([-math.pi/10, math.pi/10])
#         rays_d, rays_o = get_rays(fov, c2w)

#         self.x = rays_o[:, 0]
#         self.y = rays_o[:, 1]
#         self.view = torch.atan2(rays_d[:, 1], rays_d[:, 0])
#         x, y, has_sol = circle_line_intersect(rays_o, rays_o + rays_d, torch.tensor([0, 0]), 1)
#         self.has_sol = has_sol
#         self.color = torch.zeros(has_sol.shape + (3,))
#         self.color[has_sol] = torch.from_numpy(curve_color(x, y)).float()
    
#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         return torch.stack([self.x[idx], self.y[idx], self.view[idx]]), self.color[idx]


class OneDCurveNewData(Dataset):
    def __init__(self, n_samples=1000) -> None:
        super().__init__()
        self.n_samples = n_samples
        c2w = sample_camera_pose(n_samples, near=2, far=4)
        fov = torch.tensor([-math.pi/18, math.pi/18])
        rays_d, rays_o = get_rays(fov, c2w)
        # import pdb; pdb.set_trace()
        

        self.x = rays_o[:, 0]
        self.y = rays_o[:, 1]
        self.rays_o = rays_o
        self.view = rays_d # torch.atan2(rays_d[:, 1], rays_d[:, 0])
        x, y, has_sol = circle_line_intersect(rays_o, rays_o + rays_d, torch.tensor([0, 0]), 1)
        self.has_sol = has_sol
        self.color = torch.zeros(has_sol.shape + (3,))
        self.color[has_sol] = torch.from_numpy(curve_color(x, y)).float()
    
    def __len__(self):
        # return self.n_samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.cat([self.rays_o[idx], self.view[idx]]), self.color[idx]



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


class OneDCurveOccData(Dataset):
    def __init__(self,n_samples=1000) -> None:
        super().__init__()

        self.n_samples = n_samples
        c2w = sample_camera_pose(n_samples, near=5, far=10)
        fov = torch.tensor([-math.pi/10, math.pi/10])
        rays_d, rays_o = get_rays(fov, c2w)

        self.x = rays_o[:, 0]
        self.y = rays_o[:, 1]
        self.view = torch.atan2(rays_d[:, 1], rays_d[:, 0])
        # self.has_sol = has_sol
        x1, y1, has_sol1 = circle_line_intersect(rays_o, rays_o + rays_d, torch.tensor([1, 0]), 1)
        x2, y2, has_sol2 = circle_line_intersect(rays_o, rays_o + rays_d, torch.tensor([-1, 0]), 1)

        # color is taken from closest points from (x1, y1) or (x2, y2)
        self.color = torch.zeros(has_sol1.shape + (3,))
        self.color[has_sol1] = torch.from_numpy(curve_color(x1, y1)).float()
        self.color[has_sol2] = torch.from_numpy(curve_color(x2, y2)).float()
        # solve conflict in overlapping area
        
        

       
        # near = 1.5 
        # far = 4
        # self.n_samples = n_samples
        # r_rand  = torch.rand(n_samples)
        # r = (1 - r_rand) * near + r_rand * far
        # theta = torch.rand(n_samples) * 2 * math.pi
        # x = r * torch.cos(theta)
        # y = r * torch.sin(theta)
        # x1 = r * (torch.cos(2* theta) + 1)
        # y1 = r * torch.sin(2 * theta)
        # view1 = theta + math.pi 
        # c1 = torch.from_numpy(curve_color(x/r, y/r))
        # x2 = -r * (torch.cos(2* theta) + 1)
        # y2 = -r * torch.sin(2 * theta)
        # view2 = theta 
        # c2 = torch.from_numpy(curve_color(x/r, -y/r))
        # # import pdb; pdb.set_trace()
        # self.x = torch.cat([x1, x2])
        # self.y = torch.cat([y1, y2])
        # self.view = torch.cat([view1, view2])
        # self.color = torch.cat([c1, c2])
        #self.color = torch.from_numpy(curve_color(self.x/r, self.y/r))
        # self.color[torch.abs(curve_fn(self.x, self.y)) > 1e-1] = 0
    
    def __len__(self):
        return self.n_samples * 2

    def __getitem__(self, idx):
        return torch.stack([self.x[idx], self.y[idx], self.view[idx]]), self.color[idx]


if __name__ == "__main__":
    from vis import vis_data
    dataset = OneDCurveOccData(10)
    import pdb; pdb.set_trace()
    vis_data(dataset.x, dataset.y, dataset.view, dataset.color)