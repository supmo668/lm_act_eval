from beartype import beartype
from playwright.sync_api import Page

from .base import PseudoPage


@beartype
def get_query_text(page: Page | PseudoPage, selector: str) -> str:
    """Get the text content of the element matching the given selector.

    Note that this function DOES NOT perform downcasing.
    """
    try:
        result = page.evaluate(
            f"""
                (() => {{
                    try {{
                        return document.querySelector('{selector}').textContent;
                    }} catch (e) {{
                        return '';
                    }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result
    
@beartype
def get_query_text_lowercase(page: Page | PseudoPage, selector: str) -> str:
    """Get the lowercase text content of the element matching the given selector."""
    return get_query_text(page, selector).lower()

class FunctionRegistry:
    registry = {}

    @classmethod
    def registery(cls, name):
        def decorator(func):
            cls.registry[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name):
        """Retrieve a function by name."""
        return cls.registry.get(name)
    
    @classmethod
    def list(cls):
        """List all registered functions."""
        return list(cls.registry.keys())

function_registry = FunctionRegistry()