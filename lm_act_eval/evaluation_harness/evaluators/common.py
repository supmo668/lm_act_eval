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

from .webarena_rl.browser_env.actions import Action
from .webarena_rl.browser_env.utils import StateInfo
from .webarena_rl import image_utils
# from .webarena_rl.helper_functions import (
#     PseudoPage,
#     get_query_text,
#     get_query_text_lowercase,
#     gitlab_get_project_memeber_role,
#     llm_fuzzy_match,
#     llm_ua_match,
#     reddit_get_latest_comment_content_by_username,
#     reddit_get_latest_comment_obj_by_username,
#     reddit_get_parent_comment_username_of_latest_comment_by_username,
#     reddit_get_post_url,
#     shopping_get_latest_order_url,
#     shopping_get_num_reviews,
#     shopping_get_order_product_name_list,
#     shopping_get_order_product_option,
#     shopping_get_order_product_quantity,
#     shopping_get_product_attributes,
#     shopping_get_product_price,
#     shopping_get_rating_as_percentage,
#     shopping_get_sku_latest_review_author,
#     shopping_get_sku_latest_review_rating,
#     shopping_get_sku_latest_review_text,
# )

from . import USER_AGENT_HEADERS

from .registry import evaluator_registry, metric_registry

# import the evauators for automatic registration

from .sft.trajectory import TableTrajectoryEvaluator
from .metrics.external import levenshtein_comparator, contextual_precision

from lm_act_eval.evaluation_harness.openai.vision.evaluator import GPTVScorer


