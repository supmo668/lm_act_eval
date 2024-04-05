import googlemaps
from datetime import datetime
import os
import time
import random

from dotenv import load_dotenv
load_dotenv()

class GooglePlacesClient:
    def __init__(self):
        """Initialize the Google Places client with a provided API key."""
        assert "GOOGLE_PLACES_API_KEY" in os.environ, "Please ensure `GOOGLE_PLACES_API_KEY` api key is available"
        self.gmaps = googlemaps.Client(key=os.getenv("GOOGLE_PLACES_API_KEY"))

    def fetch_restaurants(self, location, radius=1000):
        """Fetch restaurant names within a given radius of a location.

        Args:
            location (str): The location around which to retrieve place information.
                            This must be a string that googlemaps can interpret.
            radius (int): Distance in meters within which to search for restaurants.
                          Defaults to 1000.

        Returns:
            list: A list of restaurant names.
        """
        restaurant_names = []
        try:
            places_result = self.gmaps.places_nearby(location=location, radius=radius, type='restaurant')

            # Collect initial page of results
            restaurant_names.extend(self._extract_names(places_result))

            # Handle pagination if there are more results
            while 'next_page_token' in places_result:
                token = places_result['next_page_token']
                time.sleep(2)  # Delay for token to become valid
                places_result = self.gmaps.places_nearby(page_token=token)
                restaurant_names.extend(self._extract_names(places_result))
        except googlemaps.exceptions.ApiError as e:
            print(f"API error: {e}")
        except googlemaps.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
        except googlemaps.exceptions.Timeout as e:
            print(f"Request timed out: {e}")
        except googlemaps.exceptions.TransportError as e:
            print(f"Transport error: {e}")
        return restaurant_names

    def _extract_names(self, places_result):
        """Extract restaurant names from the places_result.

        Args:
            places_result (dict): The result from a googlemaps places request.

        Returns:
            list: A list of restaurant names extracted from the places_result.
        """
        return [place['name'] for place in places_result.get('results', [])]

        
if __name__ == "__main__":
    API_KEY = os.getenv('GOOGLE_PLACES_API_KEY', 'YOUR_API_KEY')  # Preferably set your API key as an environment variable
    client = GooglePlacesClient()
    
    # Example usage: Fetching restaurants near New York City
    restaurants = client.fetch_restaurants(location='40.7128,-74.0060')  # Latitude and longitude of New York City
    print(restaurants)
