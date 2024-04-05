from .wandb import WandbLogger
# Assume BraintrustLogger is similarly defined in braintrust_logger.py
from .braintrust import BraintrustLogger
import pandas as pd

def parse_config(config):
    if 'logging' in config:
        return config.get('logging', {})
    else:
        return config

class LoggerRouter:
    def __init__(self, config_path):
        self.loggers = {}
        config = parse_config(config_path)

        for logger_name, settings in config.items():
            if logger_name == 'wandb':
                self.loggers[logger_name] = WandbLogger(**settings)
            elif logger_name == 'braintrust':
                self.loggers[logger_name] = BraintrustLogger(**settings)
            else:
                print(f"Warning: Unsupported logger '{logger_name}' in configuration. Skipping.")

    def log(self, data: pd.DataFrame|pd.Series| list[pd.Series], logger_type, *args, **kwargs):
        if logger_type not in self.loggers:
            raise ValueError(f"Logger '{logger_type}' not initialized. Check your configuration.")
        try:
            if isinstance(data, list):
                for series in data:
                    self.loggers[logger_type].log(series, *args, **kwargs)
            else:
                self.loggers[logger_type].log(data, *args, **kwargs)
        except Exception as e:
            print(f"Error logging data with '{logger_type}': {str(e)}")

if __name__=="__main__":
    # Assuming your configuration is stored in config.yaml
    logger_router = LoggerRouter(config_path='config.yaml')

    # Example data series or DataFrame
    data_series = [pd.Series({
        'input': 'Example input', 'output': 'Example output'})]

    # Log the data to all initialized loggers
    for logger_name in logger_router.loggers.keys():
        logger_router.log(data=data_series, logger_type=logger_name, result="result_value")
