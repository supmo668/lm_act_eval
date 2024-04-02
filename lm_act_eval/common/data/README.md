# README for Data Augmentation Script

## Overview

This script provides a data augmentation framework specifically designed for anonymizing and augmenting datasets containing user context information and DOM strings related to restaurant names. Utilizing the `Faker` library, the script replaces real user information and restaurant names with fictional yet realistic data, ensuring privacy and confidentiality are maintained. The framework is highly suitable for generating synthetic datasets for development, testing, and presentation purposes without compromising personal or sensitive data.

## Features

- **User Context Augmentation**: Randomizes user addresses, emails, and names within a given string, replacing them with fictitious yet plausible alternatives.
- **Restaurant Name Augmentation**: Identifies restaurant names within DOM strings and substitutes them with random restaurant names, preserving the integrity of the DOM structure.
- **Customizable Data Fields**: Supports extension to other data fields and formats through the augmentation function dictionary, `augment_fs`.
- **Pandas Integration**: Seamlessly operates on pandas DataFrames, making it ideal for data science and machine learning workflows.

## Dependencies

This project is managed with Poetry to handle dependencies and environments. Below are the primary dependencies:

- Python 3.6+
- pandas
- Faker
- tqdm (Optional for progress bars)
- A custom utility module for extracting restaurant names, `.utils.extract_restaurant_names`

### Installing Extra Dependencies

For specific tasks such as data operations (`dataops`), additional dependencies can be installed. The required packages are defined in the `pyproject.toml` file under the `[tool.poetry.group.dataops.dependencies]` section. To install these dependencies, use the following configuration in your `pyproject.toml`:

```toml
[tool.poetry.group.dataops.dependencies]
faker = "^24.4.0"
```

To install the `dataops` group dependencies, run:

```sh
poetry install -E dataops
```

## Installation

Ensure Python 3.6 or later is installed on your system. Install the required Python packages using Poetry:

```sh
poetry install
```

Note: The custom utility module `utils` containing the `extract_restaurant_names` function must be accessible within your project's directory structure.

## Usage

The script can be executed directly from the command line or imported as a module in other Python scripts.

### Command Line Usage

To run the script on a dataset:

```sh
python script_name.py path_to_input_csv path_to_output_csv
```

Replace `script_name.py` with the actual name of the script file, `path_to_input_csv` with the path to your input CSV file containing the dataset, and `path_to_output_csv` with the desired path for the augmented dataset.

### Module Usage

Import and use the augmentation functions within your Python code:

```python
from script_name import augment_dataframe
import pandas as pd

# Load your DataFrame
df = pd.read_csv('path_to_input_csv')

# Augment DataFrame
df_augmented = df.apply(augment_dataframe, axis=1)

# Save the augmented DataFrame
df_augmented.to_csv('path_to_output_csv')
```

## Function Descriptions

- **`randomize_user_context(user_context) -> (str, dict)`**: Augments a string containing user context information, returning the augmented string and a dictionary mapping new values to the original ones.

- **`extract_and_replace_restaurant_names(dom_string) -> (str, dict)`**: Identifies and replaces restaurant names within a DOM string, returning the modified DOM string and a dictionary of the original names and their replacements.

- **`augment_dataframe(series) -> pd.Series`**: Augments user context and restaurant names within a pandas Series object, designed to be applied row-wise to a pandas DataFrame.

## Contribution

Contributions, bug reports, and feature requests are welcome. Please refer to the project's contribution guidelines for more information on how to contribute.