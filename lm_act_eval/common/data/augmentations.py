import pandas as pd
from faker import Faker
import re
import random
from tqdm import tqdm 

from .utils import extract_restaurant_names
from .googlemaps import GooglePlacesClient

# Initialize Faker with US locale for North American data
fake = Faker('en_US')


def randomize_ids_links(dom_string):
    # Extract all ids and link numbers
    ids = [int(x) for x in re.findall(r'id=(\d+)', dom_string)]
    link_numbers = [int(x) for x in re.findall(r'link_(\d+)', dom_string)]
    
    # Combine and find the maximum value
    all_numbers = ids + link_numbers
    max_value = max(all_numbers)
    
    # Generate a shuffled list of new ids/links
    new_ids = list(range(max_value + 1))
    random.shuffle(new_ids)
    
    # Mapping of old ids/links to new ones
    id_map = {old: new for old, new in zip(sorted(all_numbers), new_ids)}
    
    # Substitute ids
    for old, new in id_map.items():
        dom_string = re.sub(f'id={old}(?=[\s>])', f'id={new}', dom_string)
        dom_string = re.sub(f'link_{old}', f'link_{new}', dom_string)
    
    return dom_string
  
def randomize_user_context(user_context):
    """
    Randomize the user context by replacing specific fields with new values.

    Parameters:
    user_context (dict): The original user context dictionary.

    Returns:
    tuple: A tuple containing the updated user context dictionary and a dictionary mapping the old values to the new values.
    """
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

gmap = GooglePlacesClient()
fake = Faker()

# Function to generate random restaurant names in Northern California
def random_restaurants_norcal(
    n, lat_range=(36.0, 39.0), long_range=(-124.0, -120.0)):
    """
    A function to generate random restaurants in Northern California within specified latitude and longitude ranges.

    Parameters:
    - n (int): The number of random locations to generate.
    - lat_range (tuple): A tuple representing the latitude range (default is (36.0, 39.0)).
    - long_range (tuple): A tuple representing the longitude range (default is (-124.0, -120.0)).

    Returns:
    None
    """
    all_restaurant_names = []
    # Approximate latitude and longitude ranges for Northern California

    for _ in range(n):
        # Generate a random location within the ranges
        random_location = (random.uniform(*lat_range), random.uniform(*long_range))
        # Fetch restaurants for the random location
        restaurants = gmap.fetch_restaurants()(random_location)
        all_restaurant_names.extend(restaurants)

def extract_and_replace_restaurant_names(
    dom_string, use_google:bool=True):
    # Extract original restaurant names
    original_names = extract_restaurant_names(dom_string)
    
    # Dictionary to hold original names and their fake replacements
    name_replacements = {}
    
    if use_google:
        replacements = random_restaurants_norcal(len(original_names))
    # Replace names
    for i, original_name in enumerate(original_names):
        # Generate a fake name if we haven't already    
        if original_name not in name_replacements:
            if use_google:
                fake_name = replacements[i]
            else:
                fake_name = fake.company()
            name_replacements[original_name] = fake_name
            # Replace in the DOM string
            dom_string = dom_string.replace(original_name, fake_name)
    
    return dom_string, name_replacements
  
augment_fs = {
  'DOM': randomize_ids_links,
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
    
def main(df):
    return df.apply(augment_dataframe, axis=1)

if __name__ == "__main__":
    # Sample DataFrame
    data = {'USER_CONTEXT': ['userAddress: 9926 Tesson Creek Estates Rd; userEmail: edmund.martin.mills@gmail.com; userName: Edmund Mills; userNotes: I am vegan, never order me any foods containing animal products. If I tell you to buy something, get the purchase ready to execute, then check in with me about making the purchase;']}
    import sys
    df_path = sys.argv[1]
    df_path_augmented = sys.argv[2]
    
    df = pd.DataFrame(df_path)
    
    df_augmented = df.apply(augment_dataframe, axis=1)
    df_augmented.to_csv(df_path_augmented)