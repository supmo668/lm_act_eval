# lm_act_eval/lm_act_eval/__main__.py
import hydra
from omegaconf import DictConfig, OmegaConf

# The decorator automatically reads the configuration from the specified directory
@hydra.main(
  version_base=None, config_path="../config",
  config_name="opentable_trajectory")
def main(cfg: DictConfig) -> None:
    print("Configuration:", OmegaConf.to_yaml(cfg))
    # Your main function logic here

if __name__ == "__main__":
    main()
