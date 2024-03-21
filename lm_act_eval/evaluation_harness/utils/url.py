import requests
from typing import LiteralString
from pandas import Series, DataFrame

def is_screenshot_url_accessible(url: Series| LiteralString, field_name="screenshot"):
    if type(url) == Series | DataFrame:
        assert field_name, "Field name must be specified for which field is the url field"
        url = url.get(field_name)
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        # Fall back to GET request if HEAD is not allowed
        if response.status_code == 405:
            response = requests.get(url, stream=True, timeout=5)
        return response.status_code == 200 and any([img in response.headers.get('Content-Type', '') for img in ['image/png', 'image/jpg', 'image/jpeg']])
    except requests.RequestException:
        return False
