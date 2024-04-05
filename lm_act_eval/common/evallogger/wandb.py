import wandb
from base_logger import BaseLogger
import pandas as pd

class WandbLogger(BaseLogger):
    def __init__(self, project_name):
        wandb.init(project=project_name)

    def log(self, data, artifact_name, table_name, filename):
        if isinstance(data, pd.DataFrame):
            wandb_table = wandb.Table(dataframe=data)
        else:
            raise TypeError("Data must be a pandas DataFrame.")

        artifact = wandb.Artifact(
            artifact_name, type="dataset")
        artifact.add(wandb_table, table_name)
        artifact.add_file(filename)
        
        wandb.log({table_name: wandb_table})
        wandb.log_artifact(artifact)
