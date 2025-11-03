import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer


class TransferResNet(pl.LightningModule):
    def __init__(self, lr=1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.CrossEntropyLoss()(self(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.fc.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    val = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32)

    model = TransferResNet()
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader)
