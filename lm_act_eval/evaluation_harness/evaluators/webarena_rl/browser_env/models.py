"""_summary_
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, Union

from .actions import Action
from .utils import StateInfo

Trajectory = list[Union[StateInfo, Action]]

class Site(BaseModel):
    slug: str
    url: str
    token: Optional[str] = Field(default="")
    reset_token: Optional[str] = Field(default="")
    username: Optional[str] = Field(default="")
    password: Optional[str] = Field(default="")
    exact_match: bool = Field(default=True)
    keyword: Optional[str] = Field(default="")

class Params(BaseModel):
    sites: Dict[str, Site]

class Config(BaseModel):
    browse: dict
    params: Params
