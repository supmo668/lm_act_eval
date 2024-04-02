from omegaconf import DictConfig
from lm_act_eval.evaluation_harness.evaluators import evaluator_registry, metric_registry

def handle_sft(eval_config: DictConfig) -> None:
   for eval_track, track_conf in eval_config.items():
        match eval_track:
            case "trajectory":
                handle_sft_trajectory(track_conf)
            case _:
                raise ValueError(
                  f"Unsupported evaluation track: {eval_track}")



def handle_sft_trajectory(eval_detail: DictConfig) -> None:
    """
    Handles the SFT trajectory by creating a trajectory evaluator and evaluating it.

    Args:
        detail (DictConfig): The detail configuration for the trajectory.
    """
    traj_evaluator = evaluator_registry.get('sft.trajectory')(eval_detail)
    traj_evaluator.evaluate()