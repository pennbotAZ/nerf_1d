from model import NeRF, NeRF_Debug, NeRF_New, NeRF_Noview, NeRF_New
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloader import *
def main():
    model = NeRF()
    trainer = pl.Trainer(gpus=1)
    dataset = OneDCurveData()
    train_loader = DataLoader(dataset, batch_size=64)
    trainer.fit(model, train_loader)

def mainNew():
    # torch.manual_seed(0) 
    model = NeRF_New()
    trainer = pl.Trainer(gpus=1)
    dataset = OneDCurveNewData(500)
    torch.save(dataset, 'dataset.pth')
    print('black color', (dataset.color.sum(-1) == 0).sum())
    print('all color', (dataset.color.shape))
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    trainer.fit(model, train_loader)

def mainOcc():
    model = NeRF()
    trainer = pl.Trainer(gpus=1)
    dataset = OneDCurveOccData()
    train_loader = DataLoader(dataset, batch_size=64)
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    mainNew()
