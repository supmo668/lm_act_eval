from .wandb import WandbLogger
# Assume BraintrustLogger is similarly defined in braintrust_logger.py
from .braintrust import BraintrustLogger


class LoggerRouter:
    def __init__(self, logger_type, *args, **kwargs):
        self.logger = self._get_logger(logger_type, *args, **kwargs)

    def _get_logger(self, logger_type, *args, **kwargs):
        match logger_type:
            case 'wandb':
                return WandbLogger(*args, **kwargs)
            case 'braintrust':
                return BraintrustLogger(*args, **kwargs)
            case _:
                raise ValueError(f"Unsupported logger type: {logger_type}")

    def log(self, *args, **kwargs):
        self.logger.log(*args, **kwargs)

    def log(self, *args, **kwargs):
        self.logger.log(*args, **kwargs)
