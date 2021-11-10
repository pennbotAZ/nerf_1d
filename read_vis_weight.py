import torch
import matplotlib.pyplot as plt
import numpy as np


max_idx = 7
with torch.no_grad():
    for i in range(max_idx)):
        rgb = torch.load('rgb_{}.pth'.format(i), map_location='cpu')
        weight = torch.load('weight_{}.pth'.format(i), map_location='cpu')
        plt.imshow(rgb)
        plt.show()

with torch.no_grad():
    for i in range(max_idx):
        fig, axs = plt.subplots(10)
        # rgb = torch.load('rgb_{}.pth'.format(i))
        weight = torch.load('weight_{}.pth'.format(i), map_location='cpu')
        for j in range(10):
            axs[j].bar(np.arange(weight.shape[1]), weight[j].cpu())
        plt.show()
        print('done')