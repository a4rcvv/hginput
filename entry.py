import click
import command

import logging
import sys


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Show debug log")
def cli(verbose: bool):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        level=log_level,
    )


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


@dataset.command(help="Merge raw datasets into one file.")
@click.option("-f", "--filename")
@click.option(
    "--exclude", multiple=True, help="These labels will be excluded.", default=[]
)
def merge(filename: str, exclude: list[str]):
    command.merge(filename, tuple(exclude))


if __name__ == "__main__":
    cli()
