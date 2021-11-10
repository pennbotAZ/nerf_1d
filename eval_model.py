
from model import NeRF
import torch
from vis import render_path_circle, vis_res


# render new view in the circle
model = NeRF.load_from_checkpoint('lightning_logs/version_34_10/checkpoints/epoch=401-step=401.ckpt')
model.eval()
model.cuda()
with torch.no_grad():
    x, y, res = render_path_circle(model)
res = res.cpu()
vis_res(x, y, res)
