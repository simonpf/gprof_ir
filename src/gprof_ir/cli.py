"""
gprof_ir.cli
============

Defines the command line interface for GPROF IR.
"""
import logging

import click
from rich.logging import RichHandler

from . import retrieval
from .config import CONFIG


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


@gprof_ir.command(name="config")
def config():
    """
    Display configuration.
    """
    CONFIG.print()


gprof_ir.command(name="retrieve")(retrieval.cli)
