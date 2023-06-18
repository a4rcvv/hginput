from collections import OrderedDict
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.functional.classification.accuracy import multiclass_accuracy

# from torchmetrics.classification.accuracy import MulticlassAccuracy


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
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> Any:
        x = self.layers(x)
        return x

    def metrics(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = multiclass_accuracy(
            y_hat, torch.argmax(y, dim=1), num_classes=self.n_classes, average="micro"
        )
        return loss, acc

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        loss, acc = self.metrics(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        loss, acc = self.metrics(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        loss, acc = self.metrics(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
