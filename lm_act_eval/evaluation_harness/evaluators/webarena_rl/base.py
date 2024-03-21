
import json
from pathlib import Path
from beartype import beartype
from beartype.door import is_bearable
from playwright.sync_api import CDPSession, Page

from .browser_env.actions import Action
from .browser_env.utils import StateInfo
from evaluation_harness.helper_functions import PseudoPage

from typing import Union

from ..metrics.string import StringEvaluator
from ..metrics.numeric import NumericEvaluator
from ..metrics.image import PageImageEvaluator
from ..metrics.url import URLExactEvaluator, HTMLContentExactEvaluator

Trajectory = list[Union[Action, StateInfo]]


@beartype
class EvaluatorPartial(object):
    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage,
        client: CDPSession,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        try:
            is_bearable(trajectory[-1], Action)
            last_action = trajectory[-1]
        except Exception:
            raise ValueError(
                "The last element of trajectory should be an action, add a fake stop action if needed"
            )

        return last_action  # type: ignore[return-value]
    
@beartype
class Evaluator(EvaluatorPartial):
    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        try:
            is_bearable(trajectory[-1], Action)
            last_action = trajectory[-1]
        except Exception:
            raise ValueError(
                "The last element of trajectory should be an action, add a fake stop action if needed"
            )

        return last_action  # type: ignore[return-value]

    @staticmethod
    def get_last_state(trajectory: Trajectory) -> StateInfo:
        try:
            is_bearable(trajectory[-2], StateInfo)
            last_state = trajectory[-2]
        except Exception:
            raise ValueError(
                "The second last element of trajectory should be a state, add a fake stop action if needed"
            )

        return last_state  # type: ignore[return-value]
    
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