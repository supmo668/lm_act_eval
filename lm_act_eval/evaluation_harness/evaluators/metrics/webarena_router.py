
import json
from pathlib import Path
from beartype import beartype
from beartype.door import is_bearable
from typing import Union

from playwright.sync_api import CDPSession, Page

from lm_act_eval.evaluation_harness.helper_functions import PseudoPage
from lm_act_eval.evaluation_harness.evaluators.webarena_rl.base import EvaluatorComb, Evaluator, EvaluatorPartial


from .string import StringEvaluator
# from ..metrics.numeric import NumericEvaluator
from .image import PageImageEvaluator
from .url import URLExactEvaluator, HTMLContentExactEvaluator


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