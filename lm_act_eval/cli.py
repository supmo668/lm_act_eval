"""Console script for lm_act_eval."""
import sys
import click

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra import compose, initialize

# Import other evaluators as needed
from . import evaluator_registry

from pathlib import Path

# relative path to root
CONFIG_STORE_rel = Path("../../../configs/evaluate")

@click.option(
    '--config-name', default='opentable_trajectory', 
    help='Name of the configuration to use'
)
def main(args=None):
    """Console script for lm_act_eval."""
    print("Registered evaluators: ", evaluator_registry.list_registered())
    click.echo("Replace this message by putting your code into "
               "lm_act_eval.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
