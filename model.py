import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn

class HackathonModel(LightningModule):

    def __init__(self):
        super(HackathonModel, self).__init__()
        self.build_model()

    def build_model(self):
        # EXAMPLE MODEL, CHANGE FOR HACKATHON
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 96, 5),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 160, 5),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.Flatten()
        )

        self.linear = nn.Linear(10240, 10)

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log('loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log('loss_val', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def _shared_step(self, batch):
        logits = self.forward(batch)

        loss = self.calc_loss(logits, batch[1])

        metrics = self.calc_metrics(logits, batch[1])

        return loss, metrics

    def forward(self, batch):
        encoding = self.encode(batch[0])

        logits = self.linear(encoding)

        return logits

    def calc_loss(self, prediction, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')

        loss = ce_loss(prediction, target)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        opt = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

        return opt

    def calc_metrics(self, logits, target):
        metrics = {}

        prediction = torch.argmax(logits, dim=1)
        batch_size = len(logits)

        metrics['accuracy'] = (prediction == target).sum() / batch_size

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
