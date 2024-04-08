```python
import re
import requests
from bs4 import BeautifulSoup

async def extract_reservation_info(base_url, availability_token, correlation_id):
    """Extracts reservation information from an OpenTable confirmation page.

    Args:
        base_url: The base URL of the OpenTable confirmation page.
        availability_token: The availability token from the URL.
        correlation_id: The correlation ID from the URL.

    Returns:
        A dictionary containing the extracted reservation information.
    """

    url = f"{base_url}?availabilityToken={availability_token}&correlationId={correlation_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")

    # Extract restaurant name
    restaurant_name = soup.find("h2", class_="V7a4S-vG4WU-").find("a").text.strip()

    # Extract reservation status
    status = soup.find("div", class_="COZIdjndMo0-").find("h1").text.strip().lower()

    # Extract number of people
    party_size_text = soup.find("section", class_="_4F2eiBjq1Mg-", data-test="reservation-party-size").text
    party_size = int(re.search(r"\d+", party_size_text).group())

    # Extract reservation date and time
    date_time_text = soup.find("section", class_="_4F2eiBjq1Mg-", data-test="reservation-date-time").text
    date_time = re.search(r"\w{3}, \w{3} \d+ at \d+:\d+\w{2}", date_time_text).group()

    # Extract user information (first name, last name) assuming logged-in scenario 
    user_info_element = soup.find("div", class_="nLQ-7r8IvOk- yaIJ9q73X-w-")
    user_full_name = user_info_element.text.strip()
    first_name, last_name = user_full_name.split(" ", 1) 

    return {
        "restaurant_name": restaurant_name,
        "status": status,
        "party_size": party_size,
        "date_time": date_time,
        "first_name": first_name,
        "last_name": last_name,
    }
```
**Explanation:**

1. **Import Libraries:** Imports necessary libraries for web scraping and regular expressions.
2. **`extract_reservation_info` Function:**
   - Takes `base_url`, `availability_token`, and `correlation_id` as input.
   - Constructs the full confirmation page URL.
   - Sends a GET request and parses the HTML content using BeautifulSoup.
3. **Extract Restaurant Name:** Finds the `h2` element with the specific class and extracts the restaurant name from the anchor tag within it.
4. **Extract Reservation Status:** Finds the `div` element with the relevant classes and extracts the status text, converting it to lowercase for consistency.
5. **Extract Number of People:**
   - Finds the `section` element containing party size information.
   - Uses a regular expression to extract the numerical party size.
6. **Extract Date and Time:**
   - Finds the `section` element with date and time information.
   - Uses a regular expression to extract the date and time in the desired format.
7. **Extract User Information (First Name, Last Name):**
   - Assumes a logged-in scenario based on the provided HTML structure.
   - Finds the `div` element containing user information.
   - Extracts the full name and splits it into first and last names.
8. **Return Results:** Returns a dictionary containing the extracted information. 
