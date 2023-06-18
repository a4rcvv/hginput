from hginput.model.datamodule import GestureDataModule
from hginput.model.model import GestureClassifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
import torch


def train(
    data_tag: str,
    project_name: str,
    accelerator: str = "cpu",
    n_layers: int = 0,
    n_units: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.1,
    batch_size: int = 32,
    validation_rate: float = 0.2,
):
    datamodule = GestureDataModule(data_tag, batch_size, validation_rate)
    n_classes = len(datamodule.metadata.labels)
    model = GestureClassifier(n_classes, n_layers, n_units, lr, dropout)
    # compiled_model = torch.compile(model, backend="aot_eager")
    logger = WandbLogger(project=project_name, log_model=True)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=100,
        accelerator=accelerator,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3),
            RichProgressBar(),
            RichModelSummary(),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
