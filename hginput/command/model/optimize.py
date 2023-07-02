import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar

from hginput.model.datamodule import GestureDataModule
from hginput.model.model import GestureClassifier


def optimize(
    data_tag: str,
    device: str = "cpu",
    n_trials: int = 100,
):
    batch_size = 32
    validation_rate = 0.2

    datamodule = GestureDataModule(data_tag, batch_size, validation_rate, num_workers=1)

    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 0, 3)
        n_units = trial.suggest_int("n_units", 16, 128)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        n_classes = len(datamodule.metadata.labels)
        model = GestureClassifier(n_classes, n_layers, n_units, lr, dropout)
        trainer = pl.Trainer(
            max_epochs=1000,
            accelerator=device,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10),
                RichProgressBar(),
            ],
        )
        trainer.fit(model, datamodule=datamodule)
        return trainer.callback_metrics["val_loss"].item()

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///opt.db",
        load_if_exists=True,
        study_name=data_tag,
    )
    study.optimize(objective, n_trials=n_trials)
