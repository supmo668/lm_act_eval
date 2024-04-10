import pytest
import pandas as pd
from unittest.mock import MagicMock

from lm_act_eval.evaluation_harness.evaluators.metrics.opentable import opentable_reservation_html
from lm_act_eval.evaluation_harness.helper_functions import function_registry
extract_reservation_info= function_registry.get('opentable_extract_reservation_details')

# Assuming the necessary imports and classes have been defined
@pytest.fixture
def setup_data():
    # Sample HTML content similar to what might be extracted from OpenTable pages
    html_data = [
        '<div><h2 data-test="restaurant-name"><a>Grill House</a></h2><div data-test="reservation-state"><h1>Confirmed</h1></div><section data-test="reservation-party-size">Party size: 4</section><section data-test="reservation-date-time">April 4th, 2021, 7:00 PM</section><div data-test="profile-initials">EM</div></div>',
        '<div><h2 data-test="restaurant-name"><a>Burger Corner</a></h2><div data-test="reservation-state"><h1>Cancelled</h1></div><section data-test="reservation-party-size">Party size: 2</section><section data-test="reservation-date-time">April 5th, 2021, 8:00 PM</section><div data-test="profile-initials">JM</div></div>'
    ]
    df = pd.DataFrame({
        'DOM': html_data,
        'restaurant_name': ['Grill House', 'Burger Corner'],
        'status': ['confirmed', 'cancelled'],
        'num_people': [4, 2],
        'date_time': ['April 4th, 2021, 7:00 PM', 'April 5th, 2021, 8:00 PM'],
        'first_name': ['Edmund', 'John'],
        'last_name': ['Mills', 'Doe']
    })

    string_evaluator = MagicMock()
    string_evaluator.exact_match = MagicMock(return_value=True)

    config = {
        'column_pairs': [
            {'ref': 'restaurant_name', 'pred': 'html_restaurant_name'},
            {'ref': 'status', 'pred': 'html_status'}
        ]
    }

    return df, string_evaluator, config



def test_opentable_scorer(setup_data):
    df, string_evaluator, config = setup_data

    # Initialize the scorer with configuration
    scorer = opentable_reservation_html(config)
    scorer.str_evaluator = string_evaluator  # Assign the mocked evaluator

    # Simulate the _process function applying extract_reservation_info
    for col_info in config['column_pairs']:
        df[f"html_{col_info['ref']}"] = df['DOM'].apply(lambda x: extract_reservation_info(x)[col_info['ref']])

    # Run the scorer
    results = scorer(df)

    # Print results for verification
    print("Test Results:", results)

    # Assertions to verify correct function calls and expected logic
    for col_info in config['column_pairs']:
        assert string_evaluator.exact_match.called, "exact_match should have been called for each column pair"
        assert isinstance(results, dict), "Results should be a dictionary"
        assert f"{col_info['ref']}_vs_html_{col_info['ref']}" in results, f"Key {col_info['ref']}_vs_html_{col_info['ref']} should be in results"
        assert results[f"{col_info['ref']}_vs_html_{col_info['ref']}"] == 1.0, "The exact match score should be 1.0 assuming mocked return is True for all comparisons"

# if __name__ == "__main__":
#     pytest.main()
