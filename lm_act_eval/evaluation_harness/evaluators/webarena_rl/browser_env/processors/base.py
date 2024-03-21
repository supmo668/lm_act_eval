
from typing import Any, TypedDict
import numpy as np

from beartype import beartype
from gymnasium import spaces
from playwright.sync_api import CDPSession, Page, ViewportSize

from ..utils import (
  AccessibilityTree,
  BrowserConfig,
  BrowserInfo,
  Observation,
  png_bytes_to_numpy,
)

from typing import TypedDict, Any
from . import  (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    IGNORED_ACTREE_PROPERTIES,
    UTTERANCE_MAX_LENGTH,
)

class ObservationProcessor:
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError


class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]
    

from .image import ImageObservationProcessor
from .text import TextObervationProcessor

class ObservationHandler:
    """Main entry point to access all observation processor"""

    def __init__(
        self,
        main_observation_type: str,
        text_observation_type: str,
        image_observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ) -> None:
        self.main_observation_type = main_observation_type
        self.text_processor = TextObervationProcessor(
            text_observation_type,
            current_viewport_only,
            viewport_size,
            captioning_fn,
        )
        self.image_processor = ImageObservationProcessor(
            image_observation_type, viewport_size
        )
        self.viewport_size = viewport_size

    @beartype
    def get_observation_space(self) -> spaces.Dict:
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )

        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )

        return spaces.Dict({
            "text": text_space, "image": image_space})

    @beartype
    def get_observation(
        self, page: Page, client: CDPSession
    ) -> dict[str, Observation]:
        text_obs = self.text_processor.process(page, client)
        image_obs, content_str = self.image_processor.process(page, client)
        if content_str != "":
            text_obs = content_str
        return {"text": text_obs, "image": image_obs}

    @beartype
    def get_observation_metadata(self) -> dict[str, ObservationMetadata]:
        return {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }

    @property
    def action_processor(self) -> ObservationProcessor:
        """Return the main processor that is associated with the action space"""
        if self.main_observation_type == "text":
            return self.text_processor
        elif self.main_observation_type == "image":
            return self.image_processor
        else:
            raise ValueError("Invalid main observation type")