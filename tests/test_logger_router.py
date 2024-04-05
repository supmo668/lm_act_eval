import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import constants
from .constants import TEST_PROJECT_NAME 

# Adjust the import path according to your project structure
from lm_act_eval.common.evallogger.router import LoggerRouter

# Fetch API keys from environment variables
BRAINTRUST_API_KEY = os.getenv("BRAINTRUST_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

def test_router_initializes_wandb_logger():
    with patch('lm_act_eval.common.evallogger.router.WandbLogger') as mock_wandb_logger:
        router = LoggerRouter('wandb', project_name=TEST_PROJECT_NAME , api_key=WANDB_API_KEY)
        mock_wandb_logger.assert_called_once_with(project_name=TEST_PROJECT_NAME , api_key=WANDB_API_KEY)
# Test for BraintrustLogger initialization
def test_router_initializes_braintrust_logger():
    with patch('lm_act_eval.common.evallogger.router.BraintrustLogger') as mock_braintrust_logger:
        router = LoggerRouter('braintrust', project_name=TEST_PROJECT_NAME, api_key=BRAINTRUST_API_KEY)
        mock_braintrust_logger.assert_called_once_with(project_name=TEST_PROJECT_NAME, api_key=BRAINTRUST_API_KEY)

# Test for log method delegation
def test_router_log_method_delegates_to_correct_logger():
    mock_logger = Mock()
    with patch('lm_act_eval.common.evallogger.router.LoggerRouter._get_logger', return_value=mock_logger):
        # Using 'mock_logger' as a placeholder logger type for this test
        router = LoggerRouter('mock_logger', project_name=TEST_PROJECT_NAME, api_key=BRAINTRUST_API_KEY)
        router.log(data="test_data")
        mock_logger.log.assert_called_once_with(data="test_data")
