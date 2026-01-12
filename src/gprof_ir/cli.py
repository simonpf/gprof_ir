"""
gprof_ir.cli
============

Defines the command line interface for GPROF IR.
"""
from importlib.metadata import version
import logging

import click
from rich.logging import RichHandler

from . import retrieval
from .config import CONFIG, set_model_path


FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
    force=True
)


LOGGER = logging.getLogger(__name__)


@click.group()
@click.version_option(version("gprof_ir"))
def gprof_ir():
    """
    The command line interface for the GPROF IR preciptiation retrieval.
    """
    pass


@gprof_ir.group()
def config():
    """
    Display and set configuration options.
    """

@config.command(name="show")
def display_config():
    """
    Display configuration.
    """
    CONFIG.print()


config.command(name="set_model_path")(set_model_path)


gprof_ir.command(name="retrieve", help="Run the GPROF IR retrieval.")(retrieval.cli_single)
gprof_ir.command(name="run", help="Run the GPROF IR retrieval.")(retrieval.cli_multi)


@gprof_ir.command(name="download_models")
@click.option(
    "--all",
    is_flag=True,
    help="Forces download of all models instead of just the default model."
)
def download_models(all: bool = True):
    """
    Download GPROF-IR models required for inference.
    """
    from .retrieval import download_model
    if all:
        download_model("cmb")
        LOGGER.info(
            "Downloaded CMB, 1-timestep model."
        )
        download_model("gmi")
        LOGGER.info(
            "Downloaded GMI, 1-timestep model."
        )
        download_model("gmi", 3)
        LOGGER.info(
            "Downloaded GMI, 3-timestep model."
        )
    else:
        download_model("gmi", 3)
        LOGGER.info(
            "Downloaded GMI, 3-timestep model."
        )
