import re
from .base import ObservationMetadata

def remove_unicode(input_string):
    # Define a regex pattern to match Unicode characters
    unicode_pattern = re.compile(r"[^\x00-\x7F]+")

    # Use the pattern to replace Unicode characters with an empty string
    cleaned_string = unicode_pattern.sub("", input_string)

    return cleaned_string


def create_empty_metadata() -> ObservationMetadata:
    return {
        "obs_nodes_info": {},
    }
