import argparse

from model_unet import HackathonModel
from dataset import HackathonDataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-c')

    args = parser.parse_args()

    val_dataset = HackathonDataset(type='validation')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=4,
                                shuffle=False)

    model = HackathonModel.load_from_checkpoint(args.ckpt)

    trainer = Trainer(accelerator='cpu',
                      # devices=1,
                     )
    result = trainer.test(model, val_dataloader)
