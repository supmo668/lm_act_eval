import pytest
import pandas as pd
import os
from hydra.experimental import initialize, compose

@pytest.fixture(scope="session")
def cfg():
    with initialize(config_path="../config", job_name="test_app"):
        config = compose(config_name="trajectory_eval")
        return config

@pytest.fixture(scope="session")
def test_data(cfg):
    # Load the full dataset
    df = pd.read_csv(cfg.sft.trajectory.data.path)
    print(df.chat_completion_messages.iloc[0][:100], df.columns)

    # Create a smaller subset of the data for testing
    test_df = df.head(100)
    test_data_path = "tests/test_data/small_test_data.csv"
    test_df.to_csv(test_data_path, index=False)

    # Update the configuration with the path to the test data
    cfg.sft.trajectory.data.path = test_data_path

    return cfg, test_data_path
