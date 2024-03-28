# lm_act_eval/lm_act_eval/__main__.py
import hydra
from omegaconf import DictConfig, OmegaConf

from lm_act_eval.evaluation_harness.evaluators import evaluator_registry, metric_registry
from lm_act_eval.evaluation_harness.handlers import (
  handle_sft
)
from .log_configs import logger

@hydra.main(version_base=None, config_path="../config", config_name="trajectory_eval")
def main(cfg: DictConfig) -> None:
    logger.info("Configuration loaded successfully.")
    print(f"Configurations:")
    print(OmegaConf.to_yaml(cfg))
    print(f"Available Evaluators: {evaluator_registry.list_registered()}")
    print(f"Available Metrics: {metric_registry.list_registered()}")

    eval_config = cfg.get('eval', None)
    if not eval_config:
        raise ValueError("Evaluation configuration is missing.")
    
    # Trajectory evaluation track
    for eval_type, conf in eval_config.items():
        match eval_type:
            case "sft":
                handle_sft(conf)
            case _:
                raise ValueError(
                  f"Unsupported evaluation type: {eval_type}")


if __name__ == "__main__":
    main()
