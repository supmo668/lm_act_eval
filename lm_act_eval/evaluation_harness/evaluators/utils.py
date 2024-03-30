import os
from lm_act_eval.evaluation_harness.constants import LOGGING_PLATFORMS

# log data
import wandb
wandb.login()

class LOGGING:
  WANDB: wandb.run
  
def _validate_and_login(config):
  if 'logging' in config.data:
    for platform, key in LOGGING_PLATFORMS.items():
      if platform in config.data.logging:
        assert os.getenv(key), f"{platform} API key must be supplied since logging is enabled"
      if platform=='wandb':
        assert os.getenv('entity'), f"Define `entity` for wandb since we're logging to organization space"
        import wandb
        wandb.login(key=os.getenv('WANDB_API_KEY')) 
        run = wandb.init(
          project='trajectory_eval', name="opentable-GPTV", 
          entity="multion-agi",
          reinit=True
        )