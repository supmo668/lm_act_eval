from dataclasses import dataclass, field
from typing import Optional, Any
import pandas as pd

from .base import DataFrameDescriptorBase

@dataclass
class opentable_html_input(DataFrameDescriptorBase):
    """
    Dataclass for specifying the structure of the input DataFrame expected by
    the opentable_reservation_html metric class.
    
    Attributes:
        DOM (pd.Series[str]): A series containing HTML content as strings.
        restaurant_name (Optional[pd.Series[str]]): Optional series containing restaurant names.
        status (Optional[pd.Series[str]]): Optional series containing reservation statuses.
        num_people (Optional[pd.Series[int]]): Optional series containing the number of people per reservation.
        date_time (Optional[pd.Series[str]]): Optional series containing date and time of reservations.
        first_name (Optional[pd.Series[str]]): Optional series containing first names.
        last_name (Optional[pd.Series[str]]): Optional series containing last names.
    """
    DOM: pd.Series
    restaurant_name: Optional[pd.Series] = None
    status: Optional[pd.Series] = None
    num_people: Optional[pd.Series] = None
    date_time: Optional[pd.Series] = None
    first_name: Optional[pd.Series] = None
    last_name: Optional[pd.Series] = None

@dataclass
class GPTVScorerInput(DataFrameDescriptorBase):
    """
    Dataclass for specifying the structure of the input DataFrame expected by
    the GPTVScorer metric class. This class outlines the required columns and their
    types, which are necessary for the GPTVScorer to function correctly.

    Attributes:
        QUERY (pd.Series[str]): A series containing query texts.
        GOAL (pd.Series[str]): A series containing goal descriptions.
        screenshot (pd.Series[Any]): A series containing URLs to screenshots or possibly other types of image identifiers.
    """
    QUERY: pd.Series
    GOAL: pd.Series
    screenshot: pd.Series
