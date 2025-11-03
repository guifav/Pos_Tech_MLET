import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

class TransferDifferentArchitecture(pl.LightningModule):
    def __init__(self, lr=1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

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
        return torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    val = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32)
    from pytorch_lightning import Trainer
    model = TransferDifferentArchitecture()
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader)
