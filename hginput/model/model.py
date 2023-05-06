from collections import OrderedDict
from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import multiclass_accuracy


class GestureClassifier(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        n_layers: int = 0,
        n_units: int = 32,
        lr: float = 1e-3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # first layer
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("linear_0", nn.Linear(64, n_units)),
                    ("relu_0", nn.ReLU()),
                    ("dropout_0", nn.Dropout(dropout)),
                ]
            )
        )
        # middle layers
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (f"linear_{i+1}", nn.Linear(n_units, n_units)),
                            (f"relu_{i+1}", nn.ReLU()),
                        ]
                    )
                )
            )
        # last layer
        self.layers.append(
            nn.Sequential(
                OrderedDict(
                    [
                        (f"linear_{n_layers+1}", nn.Linear(n_units, n_classes)),
                    ]
                )
            )
        )
        self.lr = lr

    def forward(self, x: torch.Tensor) -> Any:
        x = self.layers(x)
        return x

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = multiclass_accuracy(
            preds=torch.argmax(y_hat, dim=1),
            target=torch.argmax(y, dim=1),
            num_classes=y.shape[1],
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
