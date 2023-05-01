import click
import command

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    level=logging.INFO,
)


@click.group()
def cli():
    pass


@cli.command(
    help="Gather samples of dataset."
    "Press R to switch recording status, Press Q to quit."
)
@click.argument("label")
@click.option("--device", type=int, default=0)
@click.option("--width", type=int, default=480)
@click.option("--height", type=int, default=270)
@click.option("--fps", type=int, default=30)
def gather(label: str, device: int, width: int, height: int, fps: int):
    command.gather(label, device, width, height, fps)


cli()
