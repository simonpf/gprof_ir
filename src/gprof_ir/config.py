"""
gprof_ir.config
===============

Manages the configuration of the GPROF IR package.
"""
import logging
import os
from pathlib import Path
import toml

from appdirs import user_config_dir, user_data_dir
import click
from rich.console import Console
from rich.table import Table


LOGGER = logging.getLogger(__name__)


class ConfigManager:
    """
    Manager for the GPROF IR config.
    """
    def __init__(self):
        self.config_path = Path(user_config_dir("gprof_ir")) / "config.toml"
        if not self.config_path.parent.exists():
            self.config_path.parent.mkdir(parents=True)
        self.default_config = {
            "model_path": user_data_dir("gprof_ir")
        }
        self.load()

    def load(self) -> None:
        """
        Loads the configuration from file or returns default if not found.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = toml.load(f)
                    return None

            except (toml.TomlDecodeError, IOError):
                LOGGER.warning(
                    "Reading of GPROF IR config file ('%s') failed. Falling back to defaults."
                )
        self.config = self.default_config.copy()
        self.save()
        return self.config.copy()

    def print(self) -> str:
        """
        Print the current configuration.
        """
        table = Table(title="Current GPROF IR configuration:")

        # Add columns for key and value
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="yellow")

        # Add rows from the dictionary
        for key, value in self.config.items():
                table.add_row(str(key), str(value))

        # Print the table
        console = Console()
        console.print(table)


    def save(self):
        """Saves the current configuration to file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as output:
                toml.dump(self.config, output)
        except IOError as err:
            LOGGER.warning(f"Error saving configuration: %s.", err)

    def get(self, key, default=None):
        """Retrieves a configuration value, returning default if key is missing."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Sets a configuration value and saves it."""
        self.config[key] = value
        self.save()

    def reset(self):
        """Resets the configuration to defaults and saves it."""
        self.config = self.default_config.copy()
        self.save()

    def remove(self, key):
        """Removes a configuration key if it exists and saves the config."""
        if key in self.config:
            del self.config[key]
            self.save()


CONFIG = ConfigManager()


@click.argument("new_path", type=str)
def set_model_path(new_path: str) -> int:
    """
    Set the model path to NEW_PATH.
    """
    new_path = Path(new_path)
    if not new_path.exists() or not new_path.is_dir():
        LOGGER.error(
            "NEW_PATH must point to an existing directory. The provided path %s does not.",
            new_path
        )
        return 1
    CONFIG.set("model_path", str(new_path))
    LOGGER.info(
        "The 'model_path' was updated successfully."
    )
