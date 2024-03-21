"""base class for evaluation"""
# answer string match
import importlib
import json
import re
import time
import urllib
from pathlib import Path
from typing import Any, Optional, Tuple, Union
from urllib.parse import urljoin

import evaluate  # type: ignore[import]
import requests
from beartype import beartype
from beartype.door import is_bearable
from nltk.tokenize import word_tokenize  # type: ignore
from PIL import Image
from playwright.sync_api import CDPSession, Page

from browser_env.actions import Action
from browser_env.utils import StateInfo
from evaluation_harness import image_utils
from evaluation_harness.helper_functions import (
    PseudoPage,
    get_query_text,
    get_query_text_lowercase,
    gitlab_get_project_memeber_role,
    llm_fuzzy_match,
    llm_ua_match,
    reddit_get_latest_comment_content_by_username,
    reddit_get_latest_comment_obj_by_username,
    reddit_get_parent_comment_username_of_latest_comment_by_username,
    reddit_get_post_url,
    shopping_get_latest_order_url,
    shopping_get_num_reviews,
    shopping_get_order_product_name_list,
    shopping_get_order_product_option,
    shopping_get_order_product_quantity,
    shopping_get_product_attributes,
    shopping_get_product_price,
    shopping_get_rating_as_percentage,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
    shopping_get_sku_latest_review_text,
)

from . import USER_AGENT_HEADERS
from .webarena_rl.base import Evaluator

Trajectory = list[Union[Action, StateInfo]]


class EvaluatorComb:
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage,
        client: CDPSession,
    ) -> float:

        score = 1.0
        for evaluator in self.evaluators:
            cur_score = evaluator(
                trajectory, config_file, page, client)
            score *= cur_score

        return score


@beartype
def evaluator_router(
    config_file: Path | str, captioning_fn=None
) -> EvaluatorComb:
    """Router to get the evaluator class"""
    with open(config_file, "r") as f:
        configs = json.load(f)

    eval_types = configs["eval"]["eval_types"]
    evaluators: list[Evaluator | EvaluatorPartial] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator())
            case "url_match":
                evaluators.append(URLExactEvaluator())
            case "program_html":
                evaluators.append(HTMLContentExactEvaluator())
            case "page_image_query":
                evaluators.append(PageImageEvaluator(captioning_fn))
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)
