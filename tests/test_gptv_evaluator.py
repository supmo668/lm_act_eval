import pandas as pd
import pytest
from lm_act_eval.evaluation_harness.openai.vision.evaluator import GPTVScorer

def test_gptv_evaluator():
    # Set up test data
    df = pd.DataFrame({
        'chat_completion_messages': [
            '{"target": "Find the product XYZ description", "QUERY": "Navigate to XYZ product page"}',
            '{"target": "Check shipping options for XYZ", "QUERY": "Go to shipping information section"}'
        ],
        'screenshot': [
            'https://multion-client-screenshots.s3.us-east-2.amazonaws.com/36c2f0ba-e215-4081-96b4-ecb04c10a517_172ff25e-efdb-4002-b43a-0180f3f0eb19_screenshot.png',
            'https://multion-client-screenshots.s3.us-east-2.amazonaws.com/36c2f0ba-e215-4081-96b4-ecb04c10a517_6c8932f3-c31d-4683-aeb0-52a5bbe5a978_screenshot.png'
        ],
        'inputs': [
            'User navigates to the XYZ product page to find descriptions',
            'User scrolls through the product page to check shipping options'
        ]
    })

    # Create an instance of the Evaluator
    evaluator = GPTVScorer()

    # Run the evaluator
    evaluation_results = evaluator(df)

    # Check if the results are as expected
    assert isinstance(evaluation_results, pd.DataFrame)
    assert 'Score' in evaluation_results.columns
    assert 'Explanation' in evaluation_results.columns
    assert len(evaluation_results) > 0