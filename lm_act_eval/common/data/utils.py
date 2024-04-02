import re

def search_goto_url_terms(input_string):
    """
    if this is true the DOM id and link id number has to be augmented in pairs
    """
    # Define a regular expression pattern to match either "GOTO_URL id" or "GOTO_URL url"
    pattern = r'GOTO_URL (id|url)'

    # Search for the pattern in the input string
    if re.search(pattern, input_string):
        return True
    else:
        return False
  
def extract_restaurant_names(dom_string):
    # Compile patterns to match possible restaurant name occurrences
    patterns = [
        re.compile(r'alt="A photo of (.*?) restaurant"'),
        re.compile(r'Reserve table at (.*?) restaurant')
    ]
    
    # Set to store unique restaurant names
    restaurant_names = set()
    
    # Search for matches for each pattern and add them to the set
    for pattern in patterns:
        matches = pattern.findall(dom_string)
        for match in matches:
            # Clean the match to normalize the restaurant name
            name = match.strip()
            restaurant_names.add(name)
    
    return list(restaurant_names)