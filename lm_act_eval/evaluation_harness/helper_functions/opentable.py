import asyncio
from bs4 import BeautifulSoup
import re
import requests

from playwright.async_api import async_playwright
from playwright.sync_api import Page, sync_playwright

from .utils import function_registry

@function_registry.register('opentable_extract_reservation_details')
def extract_reservation_info(html_context: str, url_params=None):
    """Extracts reservation information from an OpenTable confirmation page.
    
    Args:
        html_context (str): Either the base URL of the OpenTable confirmation page or HTML content as a string.
        url_params (dict, optional): A dictionary of URL parameters to include in the request if html_context is a URL.

    Returns:
        dict: A dictionary containing the following information:
            - Restaurant name (str)
            - Reservation status (str)
            - Number of people in the reservation (int)
            - Date and time of the reservation (str)
            - User's first name (str)
            - User's last name (str)
    """
    # Determine if the input is a URL or HTML content
    if html_context.startswith('http://') or html_context.startswith('https://'):
        if url_params:
            url = html_context + "?" + "&".join(f"{k}={v}" for k, v in url_params.items())
        else:
            url = html_context
        response = requests.get(url)
        html_content = response.content
    else:
        html_content = html_context

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract reservation details
    restaurant_name_element = soup.find("h2", {"data-test": "restaurant-name"})
    restaurant_name = restaurant_name_element.find("a").text.strip() if restaurant_name_element else "Unknown"

    status_element = soup.find("div", {"data-test": "reservation-state"})
    reservation_status = status_element.find("h1").text.strip().lower() if status_element else "Unknown"

    party_size_element = soup.find("section", {"data-test": "reservation-party-size"})
    num_people = int(re.search(r"\d+", party_size_element.text).group()) if party_size_element and re.search(r"\d+", party_size_element.text) else 0

    date_time_element = soup.find("section", {"data-test": "reservation-date-time"})
    date_time = date_time_element.text.strip() if date_time_element else "Unknown"

    profile_header = soup.find("div", {"data-test": "profile-header"})
    if profile_header:
        user_info_text = profile_header.find("div").text.strip()
        match = re.search(r"Joined in (.*)", user_info_text)
        first_name, last_name = (user_info_text.split(" ", 1) + [""])[:2] if match else ("Unknown", "Unknown")
    else:
        first_name, last_name = "Unknown", "Unknown"

    return {
        "restaurant_name": restaurant_name,
        "status": reservation_status,
        "num_people": num_people,
        "date_time": date_time,
        "first_name": first_name,
        "last_name": last_name,
    }

async def opentable_extract_reservation_details(html_context: str | Page):
    """Extracts reservation details from an OpenTable confirmation page.

    Args:
        html_context (str or Playwright Page object): A URL to the reservation page or a Playwright Page object.

    Returns:
        dict: A dictionary containing the extracted reservation details, including:
            - restaurant_name (str): The name of the restaurant.
            - status (str): The status of the reservation (e.g., "confirmed").
            - party_size (int): The number of people in the reservation.
            - date_time (str): The date and time of the reservation.
            - first_name (str): The first name of the user.
            - last_name (str): The last name of the user.
    """
    page = None
    browser = None
    playwright = None

    # Check if html_context is a URL (str), then launch a browser and navigate to the URL
    if isinstance(html_context, str):
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)  # set headless=False if you want a browser UI
        page = await browser.new_page()
        await page.goto(html_context)
    elif hasattr(html_context, 'goto'):
        page = html_context
    else:
        raise ValueError("html_context must be either a URL string or a Playwright Page object.")

    try:
        # Extract reservation details
        restaurant_name_element = await page.query_selector("h2[data-test='restaurant-name'] a")
        restaurant_name = await restaurant_name_element.inner_text()

        status_element = await page.query_selector("div[data-test='reservation-state'] h1")
        status = (await status_element.inner_text()).strip().lower()

        party_size_element = await page.query_selector("section[data-test='reservation-party-size']")
        party_size_text = await party_size_element.inner_text()
        party_size = int(re.search(r"\d+", party_size_text).group())

        date_time_element = await page.query_selector("section[data-test='reservation-date-time']")
        date_time = (await date_time_element.inner_text()).strip()

        initials_element = await page.query_selector("div[data-test='profile-initials']")
        initials = (await initials_element.inner_text()).strip()
        first_name, last_name = initials[0], initials[1]

        # Return extracted details
        return {
            "restaurant_name": restaurant_name,
            "status": status,
            "party_size": party_size,
            "date_time": date_time,
            "first_name": first_name,
            "last_name": last_name
        }
    finally:
        # Close page and browser if they were opened in this function
        if page:
            await page.close()
        if browser:
            await browser.close()

    
if __name__=="__main__":
# Example usage:
  details = asyncio.run(opentable_extract_reservation_details("https://www.opentable.com/confirmation-page"))
