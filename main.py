from model import NeRF
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloader import OneDCurveData
def main():
    model = NeRF()
    trainer = pl.Trainer(gpus=1)
    dataset = OneDCurveData()
    train_loader = DataLoader(dataset, batch_size=64)
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
