import pandas as pd
from faker import Faker
import re
import random
from tqdm import tqdm 

from .utils import extract_restaurant_names
# Initialize Faker with US locale for North American data
fake = Faker('en_US')


def randomize_user_context(user_context):
    # Dictionary to store the fields to be replaced and their new values
    replacements = {
        'userAddress': fake.address().replace("\n", ", "),
        'userEmail': fake.email(),
        'userName': fake.name(),
    }
    # Perform the replacements and gather the old values
    replacement_dict = {}
    updated_context = user_context
    for key, new_value in replacements.items():
        # Use a regex pattern to match the current value and replace it
        pattern = rf'({key}: )(.+?);'
        def repl(match):
            old_value = match.group(2)
            replacement_dict[new_value] = old_value  # Map new value to old value
            return f'{key}: {new_value};'
        updated_context = re.sub(pattern, repl, updated_context)
    
    return updated_context, replacement_dict



def extract_and_replace_restaurant_names(dom_string):
    fake = Faker()
    # Extract original restaurant names
    original_names = extract_restaurant_names(dom_string)
    
    # Dictionary to hold original names and their fake replacements
    name_replacements = {}
    
    # Replace names
    for original_name in original_names:
        # Generate a fake name if we haven't already
        if original_name not in name_replacements:
            fake_name = fake.company()
            name_replacements[original_name] = fake_name
            # Replace in the DOM string
            dom_string = dom_string.replace(original_name, fake_name)
    
    return dom_string, name_replacements
  
augment_fs = {
  'USER_CONTEXT': randomize_user_context,
  'DOM': extract_and_replace_restaurant_names
}

def augment_dataframe(series):
  aug_series = pd.Series()
  for key, func in augment_fs.items():
    aug_series, rep_info = func(series.get(key))
    aug_series[key] = aug_series
    aug_series[key+'-aug_info'] = rep_info
    return aug_series
    
  
if __name__ == "__main__":
    # Sample DataFrame
    data = {'USER_CONTEXT': ['userAddress: 9926 Tesson Creek Estates Rd; userEmail: edmund.martin.mills@gmail.com; userName: Edmund Mills; userNotes: I am vegan, never order me any foods containing animal products. If I tell you to buy something, get the purchase ready to execute, then check in with me about making the purchase;']}
    import sys
    df_path = sys.argv[1]
    df_path_augmented = sys.argv[2]
    
    df = pd.DataFrame(df_path)
    
    df_augmented = df.apply(augment_dataframe, axis=1)
    df_augmented.to_csv(df_path_augmented)