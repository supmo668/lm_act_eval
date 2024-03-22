import pandas as pd
from datasets import Dataset

def dataframe_to_dataset(dataframe_path, input_text_column):
  df = pd.read_csv(dataframe_path)
  dataset = Dataset.from_pandas(df)
  return dataset, df