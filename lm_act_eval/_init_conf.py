import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

import click
# Ensure GlobalHydra is initialized only once to avoid warnings
GlobalHydra.instance().clear()
initialize(config_path="../config")

@click.command()
@click.argument('name', required=False, default='trajectory_eval-dev')
def init_conf(name):
    """
    Initializes and prints the configuration based on the provided name.
    """
    # Compose the configuration similarly to how it's done in the main script
    cfg = compose(config_name=name)    
    # Now, `cfg` is loaded and can be used just like in the main script
    print("Composed Configurations:")
    print(OmegaConf.to_yaml(cfg))

    # From here, you can directly import and call any functions/modules as needed for debugging
    # For example, to debug `handle_sft` if you have such a function
    # from lm_act_eval.evaluation_harness.handlers import handle_sft
    # handle_sft(cfg)
    return cfg

if __name__ == "__main__":
    init_conf()  # Invokes the command line interface