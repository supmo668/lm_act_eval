"""
Implements helper functions to assist evaluation cases where other evaluators are not suitable.
"""
from typing import Any
from playwright.sync_api import Page

class PseudoPage:
    def __init__(self, original_page: Page, url: str):
        """
        Initialize the class with the original_page and url.

        Parameters:
            original_page (Page): The original page object.
            url (str): The URL for the page.

        Returns:
            None
        """
        self.url = url
        self.original_page = original_page

    def __getattr__(self, attr: str) -> Any:
        # Delegate attribute access to the original page object
        if attr not in ["url"]:
            return getattr(self.original_page, attr)
        else:
            return getattr(self, attr)