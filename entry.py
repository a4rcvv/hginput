import click
from hginput import command
import torch
import torch.backends.mps

import logging
import sys
import pytorch_lightning as pl

pl.seed_everything(42)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Show debug log")
def cli(verbose: bool):
    logger = logging.getLogger("hginput")
    log_level = logging.DEBUG if verbose else logging.INFO
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    )

    logger.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)


@cli.group(help="Dataset utilities")
def dataset():
    pass


@dataset.command(
    help="Gather samples of dataset."
    "Press R to switch recording status, Press Q to quit."
)
@click.option("-l", "--label")
@click.option("--device", type=int, default=0)
@click.option("--width", type=int, default=480)
@click.option("--height", type=int, default=270)
@click.option("--fps", type=int, default=30)
def gather(label: str, device: int, width: int, height: int, fps: int):
    command.gather(label, device, width, height, fps)


@dataset.command(help="Create a dataset from raw datasets.")
@click.option("-t", "--tag", required=True, help="Tag of the dataset.")
@click.option(
    "--exclude", multiple=True, help="These labels will be excluded.", default=[]
)
def create(tag: str, exclude: list[str]):
    command.create(tag, tuple(exclude))


@cli.group(help="Model utilities")
def model():
    pass


@model.command(help="Train a model.")
@click.option("-t", "--tag", required=True, help="Tag of the dataset.")
@click.option("-p", "--project", required=True, help="Project name of wandb.")
@click.option("--layers", type=int, default=0, help="Number of middle layers.")
@click.option("--units", type=int, default=32, help="Number of middle layer units.")
@click.option("--lr", type=float, default=1e-3, help="Learning rate.")
@click.option("--dropout", type=float, default=0.1, help="Dropout rate.")
@click.option("--batch-size", type=int, default=32, help="Batch size.")
@click.option("--validation-rate", type=float, default=0.2, help="Validation rate.")
@click.option(
    "--cpu", is_flag=True, help="Use CPU instead of any accelerators(CUDA, MPS)."
)
def train(
    tag: str,
    project: str,
    layers: int,
    units: int,
    lr: float,
    dropout: float,
    batch_size: int,
    validation_rate: float,
    cpu: bool,
):
    accelerator = (
        "cpu"
        if cpu
        else "mps"
        if torch.backends.mps.is_available()
        else "gpu"
        if torch.cuda.is_available()
        else "cpu"
    )
    command.train(
        tag,
        project,
        accelerator=accelerator,
        n_layers=layers,
        n_units=units,
        lr=lr,
        dropout=dropout,
        batch_size=batch_size,
        validation_rate=validation_rate,
    )

@model.command(help="Show the structure of a model.")
def summary():
    command.summary()


if __name__ == "__main__":
    cli()
