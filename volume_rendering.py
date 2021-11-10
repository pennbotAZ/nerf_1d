"""
Rendering the volumne, i.e. 
(x, y, theta) -> color mapping 
"""
import torch 
import matplotlib.pyplot as plt
import math
from data_gen import curve_color
from vis import *

model = NeRF.load_from_checkpoint('lightning_logs/version_90/checkpoints/epoch=68-step=10832.ckpt')
model.eval()
model.cuda()
thetas = torch.linspace(-math.pi, math.pi, 360)
colors = []
for theta in thetas:
    theta = theta[None, ...]
    x = torch.cos(theta)
    y = torch.sin(theta)
    view = torch.linspace(theta[0] + math.pi/2, theta[0]+ math.pi/2*3, 180)
    color = curve_color(x, y)
    data = torch.stack([x.expand(view.shape[0]), y.expand(view.shape[0]), view]).t()
    res = torch.sigmoid(model.forward(data.cuda())[:, :3])
    colors.append(res)

thetas = torch.linspace(-math.pi, math.pi, 360)
for i, theta in enumerate(thetas):
    theta = theta[None, ...]
    view = torch.linspace(theta[0] + math.pi/2, theta[0]+ math.pi/2*3, 180)
    theta = theta.expand(view.shape[0])
    plt.scatter(theta, view, c=colors[i].cpu().detach())