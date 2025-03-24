"""
gprof_ir.cli
============

Defines the command line interface for GPROF IR.
"""
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


@click.group()
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


gprof_ir.command(name="retrieve", help="Run the GPROF IR retrieval.")(retrieval.cli)
