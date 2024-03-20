import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from beartype import beartype
from PIL import Image
from playwright.sync_api import CDPSession, Page

from evaluation_harness import image_utils
from evaluation_harness.helper_functions import (
    PseudoPage
)

from . import USER_AGENT_HEADERS
from .base import Evaluator, Trajectory

@beartype
class PageImageEvaluator(Evaluator):
    """Check whether the answer is correct by querying a vision model."""

    def __init__(self, captioning_fn):
        self.captioning_fn = captioning_fn
        # Default to 0.8 as the threshold for similarity to account for compression, resizing, etc
        # This might be too generous but we bias towards minimizing false negatives.
        self.ssim_threshold = 0.8

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage | None = None,
        client: CDPSession | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        for query in configs["eval"]["page_image_query"]:
            locator: str = query["eval_image_class"]
            target_url: str = query["eval_image_url"]
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = eval(func)

            # navigate to that url
            if target_url != "last":
                page.goto(target_url)
                time.sleep(3)  # TODO(jykoh): fix this hard-coded sleep

            # empty, use the full page
            if not locator.strip():
                images = page.get_by_role("img").all()
            # use JS to select the element
            elif locator.startswith("."):
                # Get all img children under the locator
                elements = page.query_selector_all(locator)
                images = []
                for element in elements:
                    is_img = element.evaluate(
                        'element => element.tagName === "IMG"'
                    )
                    if is_img:
                        images.append(element)
                    else:
                        images.extend(element.query_selector_all("img"))
            else:
                raise ValueError(f"Unknown locator: {locator}")

            if images == []:
                return 0.0

            all_image_pixels = []
            for image in images:
                try:
                    # Get image from URL.
                    image_url = image.get_attribute("src")
                    if not image_url.startswith(
                        ("http://", "https://", "www.")
                    ):
                        image_url = urljoin(page.url, image_url)
                    image = Image.open(
                        requests.get(
                            image_url, headers=USER_AGENT_HEADERS, stream=True).raw
                    )
                    all_image_pixels.append(image)
                except Exception as e:
                    print("[WARNING]: ", e)

            score = 1.0
            if all_image_pixels == []:
                return 0.0
            else:
                # Run the VQA eval on the image elements.
                eval_vqas = query.get("eval_vqa", [])
                assert (
                    len(eval_vqas) > 0 or "eval_fuzzy_image_match" in query
                ), "eval_vqa must have at least 2 questions or eval_fuzzy_image_match must be True"
                for qa in eval_vqas:
                    question, answer = qa["question"], qa["answer"]
                    prompt = f"Q: {question} A:"
                    pred_ans = self.captioning_fn(
                        all_image_pixels, [prompt] * len(all_image_pixels)
                    )
                    score *= float(
                        any(
                            [answer.lower() in ans.lower() for ans in pred_ans]
                        )
                    )

                if "eval_fuzzy_image_match" in query:
                    ssim_threshold = query.get(
                        "ssim_threshold", self.ssim_threshold
                    )
                    exact_match_imgs = query["eval_fuzzy_image_match"].split(
                        " |OR| "
                    )
                    all_exact_match_pixels = []

                    for exact_match_img in exact_match_imgs:
                        if exact_match_img.startswith("http"):
                            exact_match_pixels = Image.open(
                                requests.get(exact_match_img, stream=True).raw
                            )
                        else:
                            exact_match_pixels = Image.open(exact_match_img)
                        all_exact_match_pixels.append(exact_match_pixels)

                    # Check if any of the images on the page match
                    found_exact_match = False
                    for exact_match_pixels in all_exact_match_pixels:
                        for image_pixels in all_image_pixels:
                            ssim = image_utils.get_image_ssim(
                                image_pixels, exact_match_pixels
                            )
                            if ssim > ssim_threshold:
                                found_exact_match = True
                                break
                    score *= float(found_exact_match)

        return score