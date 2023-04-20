import click
import app


@click.group()
def cli():
    pass


@cli.command()
@click.argument("label")
@click.option("--device", type=int, default=0)
@click.option("--width", type=int, default=480)
@click.option("--height", type=int, default=270)
@click.option("--fps", type=int, default=30)
def gather(label: str, device: int, width: int, height: int, fps: int):
    app.gather(label, device, width, height, fps)


cli()
