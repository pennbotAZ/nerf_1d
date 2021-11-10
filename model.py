import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from run_nerf import *
class NeRF(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.relu = nn.functional.relu
        self.xyz_encoding_dims = 2
        self.viewdir_encoding_dims = 1
        self.hidden_size = hidden_size = 128
        self.layer1 = nn.Linear(self.xyz_encoding_dims, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3_1 = nn.Linear(hidden_size, 1)
        self.layer3_2 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layer6 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x, view = x[..., :-1], x[..., -1:]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        c = self.layer6(x)
        raw = torch.cat((c, sigma), dim=-1)
        return raw

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y = y.float()
        # import pdb; pdb.set_trace()
        rays_d = x[:, 2]
        rays_o = x[:, :2]
        view_dirs = rays_d
        y_hat = render_rays(rays_o, rays_d, view_dirs, self.forward)
        # import pdb; pdb.set_trace()
        loss = F.mse_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer


class NeRF_New(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.relu = nn.functional.relu
        self.xyz_encoding_dims = 2
        self.viewdir_encoding_dims = 2
        self.hidden_size = hidden_size = 128
        self.layer1 = nn.Linear(self.xyz_encoding_dims, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3_1 = nn.Linear(hidden_size, 1)
        self.layer3_2 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        self.layer5 = nn.Linear(hidden_size, hidden_size//2)
        self.layer6 = nn.Linear(hidden_size//2, 3)

    def forward(self, x):
        x, view = x[..., :-2], x[..., -2:]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        c = self.layer6(x)
        raw = torch.cat((c, sigma), dim=-1)
        return raw

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y = y.float()
        # import pdb; pdb.set_trace()
        rays_d = x[:, 2:]
        rays_o = x[:, :2]
        view_dirs = rays_d
        y_hat = render_rays(rays_o, rays_d, view_dirs, self.forward)
        # import pdb; pdb.set_trace()
        loss = F.mse_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer


class NeRF_Noview(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.relu = nn.functional.relu
        self.xyz_encoding_dims = 2
        self.viewdir_encoding_dims = 1
        self.hidden_size = hidden_size = 128
        self.layer1 = nn.Linear(self.xyz_encoding_dims, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3_1 = nn.Linear(hidden_size, 1)
        self.layer3_2 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(
            hidden_size, hidden_size
        )
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layer6 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x, view = x[..., :-1], x[..., -1:]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = feat #torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        c = self.layer6(x)
        raw = torch.cat((c, sigma), dim=-1)
        return raw

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y = y.float()
        # import pdb; pdb.set_trace()
        rays_d = x[:, 2]
        rays_o = x[:, :2]
        view_dirs = rays_d
        y_hat = render_rays(rays_o, rays_d, view_dirs, self.forward)
        # import pdb; pdb.set_trace()
        loss = F.mse_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, betas=(0.9, 0.999))
        return optimizer


class NeRF_Debug(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.relu = nn.functional.relu
        self.xyz_encoding_dims = 2
        self.viewdir_encoding_dims = 1
        self.hidden_size = hidden_size = 128
        self.layer1 = nn.Linear(self.xyz_encoding_dims, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3_1 = nn.Linear(hidden_size, 1)
        self.layer3_2 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layer6 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x, view = x[..., :-1], x[..., -1:]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        c = self.layer6(x)
        raw = torch.cat((c, sigma), dim=-1)
        return raw

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y = y.float()
        # import pdb; pdb.set_trace()
        rays_d = x[:, 2]
        rays_o = x[:, :2]
        view_dirs = rays_d
        raw, z_vals = render_rays_step(rays_o, rays_d, view_dirs, self.forward)
        # import pdb; pdb.set_trace()
        y_hat, weight, rgb = raw2outputs_step(raw, z_vals)
        step_size = 1000
        if self.global_step % step_size == 0:
            torch.save(batch, 'batch_{}.pth'.format(self.global_step // step_size))
            torch.save(weight, 'weight_{}.pth'.format(self.global_step // step_size))
            torch.save(rgb, 'rgb_{}.pth'.format(self.global_step // step_size))
        # import pdb; pdb.set_trace()
        loss = F.mse_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, betas=(0.9, 0.999))
        return optimizer
