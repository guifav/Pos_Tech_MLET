import torch
import torch.nn as nn
import pytorch_lightning as pl


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, scaling=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.02)
        self.r = r
        self.scaling = scaling

    def forward(self, x):
        original = torch.matmul(x, self.weight.t())
        lora_update = x @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return original + lora_update

class LoRATransformer(pl.LightningModule):
    def __init__(self, input_dim=64, hidden_dim=128, vocab_size=1000, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lora_linear = LoRALinear(input_dim, hidden_dim, r=4)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.lora_linear(x)
        x = self.activation(x)
        return self.output(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from pytorch_lightning import Trainer
    dataset_size, seq_length, batch_size = 500, 10, 32
    X = torch.randint(0, 1000, (dataset_size, seq_length))
    y = torch.randint(0, 1000, (dataset_size, seq_length))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    model = LoRATransformer()
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, dataloader)
