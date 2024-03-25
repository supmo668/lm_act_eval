# lm_act_eval/lm_act_eval/__main__.py
import hydra
from omegaconf import DictConfig, OmegaConf

from lm_act_eval.evaluation_harness.evaluators import evaluator_registry
from .log_configs import logger
# The decorator automatically reads the configuration from the specified directory
@hydra.main(
  version_base=None, 
  config_path="../config",
  config_name="opentable_trajectory")
def main(cfg: DictConfig) -> None:
  conf = OmegaConf.to_yaml(cfg)
  print("Configuration:\n", conf)
  print(f"Modules Available Evaluators:{evaluator_registry.list_registered()}")
  # Your main function logic here
  match conf.format:
    case "csv" | "sft-off":
      # simply dataset 
      logger.info("TODO: add csv support")
      evaluators.sft.CSVTrajectoryEvaluator
      
      pass
    case "sft-on":
      # + require model as input
      raise NotImplementedError("TODO: add sft-on support")
    case "rl":
      # + require model as input
      raise NotImplementedError("TODO: add rl support")

if __name__ == "__main__":
    main()
    