import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class bottleneck_layer(nn.Module):
    def __init__(self, ks, ch_out, ch_in, s, t):
        super(bottleneck_layer, self).__init__()
        self.res = s==1 and ch_in == ch_out

        ch_expand = int(round(t * ch_in))
        self.conv = nn.Sequential(
            ConvBNReLU(ch_in, ch_expand, kernel_size=1, stride=1),
            ConvBNReLU(ch_expand, ch_expand, kernel_size=ks, stride=s, groups=ch_expand),
            nn.Conv2d(ch_expand, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        if self.res:
            return x + self.conv(x)
        else:
            return self.conv(x)


class HackathonModel(LightningModule):

    def __init__(self):
        super(HackathonModel, self).__init__()
        self.layers = nn.ModuleList()

        self.ch_in=32

        self.config = [
            [
                [3, 16, 1, 1]
            ],
            [
                [3, 24, 2, 6],
                [3, 24, 1, 6]
            ],
            [
                [5, 40, 2, 6],
                [5, 40, 1, 6]
            ],
            [
                [3, 80, 2, 6],
                [3, 80, 1, 6],
                [3, 80, 1, 6],
                [5, 112, 1, 6],
                [5, 112, 1, 6],
                [5, 112, 1, 6]
            ],
            [
                [5, 192, 2, 6],
                [5, 192, 1, 6],
                [5, 192, 1, 6],
                [5, 192, 1, 6],
                [3, 320, 1, 6]
            ]
        ]

        self.build_model()


    def build_model(self):

        self.layers.append(nn.Conv2d(3, self.ch_in, kernel_size=3, stride=2, padding=1))

        for layer_sequence in self.config:
            layer = []
            for l in layer_sequence:
                ks, ch_out, s, t = l
                layer.append(bottleneck_layer(ks, ch_out, self.ch_in, s, t))
                self.ch_in = ch_out
            self.layers.append(nn.Sequential(*layer))

        self.layers.append(nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0))

        self.pool = nn.MaxPool2d(kernel_size=8)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1*1*1280, 128)
        self.head = nn.Linear(128, 1)

        self.relu = nn.ReLU6()

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
        output = self.forward(batch)

        loss = self.calc_loss(output, batch['class'])

        metrics = self.calc_metrics(output, batch['class'])

        return loss, metrics

    def forward(self, batch):
        h = self.layers[0](torch.transpose(batch['img'], -1, 1))

        for layer in self.layers[1:-1]:
            h = layer(h)

        h = self.layers[-1](h)

        h = self.pool(h)

        encoding = self.flatten(h)
        output = self.head(self.relu(self.linear(encoding)))
        output = torch.squeeze(output)

        return output

    def calc_loss(self, prediction, target):
        ce_loss = nn.BCELoss(reduction='none')
        m = nn.Sigmoid()

        loss = ce_loss(m(prediction), target.float())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        #lr_scheduler = torch.optim.lr_scheduler.CossineAnnealingLR(optimizer, T_max=10)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

        opt = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

        return opt

    def calc_metrics(self, prediction, target):
        metrics = {}

        prediction = prediction > 0
        batch_size = len(prediction)

        metrics['accuracy'] = (prediction == target).sum() / batch_size

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
