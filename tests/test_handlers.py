# tests/test_handlers.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from lm_act_eval.evaluation_harness.handlers import handle_sft, handle_sft_trajectory
from lm_act_eval.evaluation_harness.evaluators import evaluator_registry

@pytest.fixture
def mock_evaluator():
    """
    A fixture that returns a MockEvaluator class instance used for mocking evaluation.
    """
    class MockEvaluator:
        def evaluate(self):
            return "Evaluated"
    
    return MockEvaluator()

@pytest.fixture
def mock_registry(mock_evaluator):
    with patch('lm_act_eval.evaluation_harness.evaluators.evaluator_registry.get', return_value=mock_evaluator) as mock:
        yield mock

@pytest.fixture
def trajectory_config():
    return OmegaConf.create({"trajectory": OmegaConf.create({"param1": "value1", "param2": "value2"})})

def test_initialize_data_from_conf(test_data):
    cfg, data_path = test_data
    # Your test code here using the cfg and data_path
    assert Path(data_path).exists()

def test_handle_sft_trajectory_calls_evaluate(mock_registry, trajectory_config):
    handle_sft_trajectory(trajectory_config['trajectory'])
    mock_registry.assert_called_once_with('sft.trajectory')
    mock_registry.return_value().evaluate.assert_called_once()

def test_handle_sft_with_trajectory(mock_registry, trajectory_config):
    with pytest.raises(ValueError) as exc_info:
        handle_sft(OmegaConf.create({"invalid_track": {}}))
    assert "Unsupported evaluation track" in str(exc_info.value)
    
    handle_sft(trajectory_config)
    mock_registry.assert_called_with('sft.trajectory')
    mock_registry.return_value().evaluate.assert_called_once()

def test_handle_sft_invalid_track():
    invalid_config = OmegaConf.create({"invalid_track": {}})
    with pytest.raises(ValueError) as exc_info:
        handle_sft(invalid_config)
    assert "Unsupported evaluation track: invalid_track" in str(exc_info.value)
