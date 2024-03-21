import json
from typing import Any
from urllib.parse import urljoin
import numpy as np
from collections import defaultdict
import requests
import re
from beartype import beartype
from PIL import Image
from playwright.sync_api import CDPSession, Page, ViewportSize

from ..constants import IGNORED_ACTREE_PROPERTIES
from .base import ObservationProcessor
from .utils import create_empty_metadata, remove_unicode
from .base import AccessibilityTree
from .base import (
    IGNORED_ACTREE_PROPERTIES, 
    BrowserConfig, BrowserInfo
)

class TextObervationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ):
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size
        self.observation_tag = "text"
        self.meta_data = (
            create_empty_metadata()
        )  # use the store meta data of this observation type

        if self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            self.captioning_fn = captioning_fn
            # Cache captions.
            self.url2caption = {}

    @beartype
    def fetch_browser_info(
        self,
        page: Page,
        client: CDPSession,
    ) -> BrowserInfo:
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    @beartype
    @staticmethod
    def partially_in_viewport(
        bound: list[float], config: BrowserConfig
    ) -> bool:
        [x, y, width, height] = bound
        elem_left_bound = x
        elem_top_bound = y
        elem_right_bound = x + width
        elem_lower_bound = y + height

        not_in_viewport = (
            elem_left_bound < config["win_right_bound"]
            and elem_right_bound >= config["win_left_bound"]
            and elem_top_bound < config["win_lower_bound"]
            and elem_lower_bound >= config["win_upper_bound"]
        )
        return not_in_viewport

    @beartype
    def retrieve_viewport_info(self, info: BrowserInfo) -> None:
        """Add viewport related information to the DOMTree
        1. add union bound, which is a union of all the bounds of the nodes in the subtree
        This is only used when current_viewport_only is enabled since it is quite slow
        """
        tree = info["DOMTree"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        parent = nodes["parentIndex"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        bounds = layout["bounds"]

        graph = defaultdict(lambda: [])
        assert len(node_names) == len(parent)
        for node_idx in range(len(node_names)):
            parent_idx = parent[node_idx]
            if parent_idx != -1:
                graph[parent_idx].append(node_idx)

        union_bounds: list[list[float] | None] = [None for _ in bounds]

        def valid_bbox(bound: list[float] | None) -> bool:
            if bound is None:
                return False
            # no width or height
            if np.isclose(bound[2], 0):
                return False
            if np.isclose(bound[3], 0):
                return False
            return True

        def add_union_bound(idx: int) -> list[float] | None:
            if idx in layout_node_cursor:
                cursor = layout_node_cursor.index(idx)
                node_bound = bounds[cursor].copy()
                tree_bounds: list[Any] = [node_bound]
                for child_idx in graph[idx]:
                    child_bound = add_union_bound(child_idx)
                    tree_bounds.append(
                        child_bound.copy() if child_bound else None
                    )

                tree_bounds = [b for b in tree_bounds if valid_bbox(b)]
                # convert to absolute coordinates
                for i in range(len(tree_bounds)):
                    tree_bounds[i][2] = tree_bounds[i][0] + tree_bounds[i][2]
                    tree_bounds[i][3] = tree_bounds[i][1] + tree_bounds[i][3]

                if len(tree_bounds) == 0:
                    assert not valid_bbox(node_bound)
                    node_union_bound = [0.0, 0.0, 0.0, 0.0]
                else:
                    left_bound = min([b[0] for b in tree_bounds])
                    top_bound = min([b[1] for b in tree_bounds])
                    right_bound = max([b[2] for b in tree_bounds])
                    bottom_bound = max([b[3] for b in tree_bounds])
                    node_union_bound = [
                        left_bound,
                        top_bound,
                        right_bound - left_bound,
                        bottom_bound - top_bound,
                    ]

                # update the list
                union_bounds[cursor] = node_union_bound
            else:
                node_union_bound = None

            return node_union_bound

        add_union_bound(0)
        info["DOMTree"]["documents"][0]["layout"]["unionBounds"] = union_bounds

    @beartype
    def current_viewport_html(self, info: BrowserInfo) -> str:
        # adopted from [natbot](https://github.com/nat/natbot)
        tree = info["DOMTree"]
        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        attributes = nodes["attributes"]
        node_value = nodes["nodeValue"]
        parent = nodes["parentIndex"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        union_bounds = layout["unionBounds"]

        graph = defaultdict(lambda: [])
        for node_idx in range(len(node_names)):
            parent_idx = parent[node_idx]
            if parent_idx != -1:
                graph[parent_idx].append(node_idx)

        def dfs(idx: int) -> str:
            node_name = strings[node_names[idx]].lower().strip()
            can_skip = "#" in node_name or "::" in node_name

            inner_text = ""
            node_value_idx = node_value[idx]
            if node_value_idx >= 0 and node_value_idx < len(strings):
                inner_text = " ".join(strings[node_value_idx].split())
            node_attributes = [strings[i] for i in attributes[idx]]
            node_attributes_str = ""
            for i in range(0, len(node_attributes), 2):
                a = node_attributes[i]
                b = node_attributes[i + 1]
                b = " ".join(b.split())
                node_attributes_str += f'{a}="{b}" '
            node_attributes_str = node_attributes_str.strip()

            html = ""
            if not can_skip:
                html += f"<{node_name}"
                if {node_attributes_str}:
                    html += f" {node_attributes_str}"
                html += f">{inner_text}"
            else:
                html += f"{inner_text}"

            for child_idx in graph[idx]:
                if child_idx in layout_node_cursor:
                    cursor = layout_node_cursor.index(child_idx)
                    union_bound = union_bounds[cursor]
                    if not self.partially_in_viewport(
                        union_bound, info["config"]
                    ):
                        continue
                    html += dfs(child_idx)

            if not can_skip:
                html += f"</{node_name}>"

            return html

        html = dfs(0)
        return html

    @beartype
    def fetch_page_accessibility_tree(
        self, info: BrowserInfo, client: CDPSession
    ) -> AccessibilityTree:
        accessibility_tree: AccessibilityTree = client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]

        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        # add the bounding box of each node
        tree = info["DOMTree"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        backend_node_id = nodes["backendNodeId"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        bounds = layout["bounds"]
        union_bounds = layout["unionBounds"]
        offsetrect_bounds = layout["offsetRects"]
        backend_id_to_bound = {}

        # get the mapping between backend node id and bounding box
        for idx in range(len(node_names)):
            if idx not in layout_node_cursor:
                continue
            cursor = layout_node_cursor.index(idx)
            node_bound = bounds[cursor]
            node_union_bound = union_bounds[cursor]
            node_offsetrect_bound = offsetrect_bounds[cursor]
            node_backend_id = backend_node_id[idx]
            backend_id_to_bound[node_backend_id] = [
                node_bound,
                node_union_bound,
                node_offsetrect_bound,
            ]

        parent_graph: dict[str, str] = {}
        refine_node_ids: list[str] = []
        for node in accessibility_tree:
            if "parentId" in node:
                parent_graph[node["nodeId"]] = node["parentId"]
            if "backendDOMNodeId" not in node:
                node["bound"] = None
                node["union_bound"] = None
                node["offsetrect_bound"] = None
            elif node["backendDOMNodeId"] not in backend_id_to_bound:
                refine_node_ids.append(node["nodeId"])
            else:
                node["bound"] = backend_id_to_bound[node["backendDOMNodeId"]][
                    0
                ]
                node["union_bound"] = backend_id_to_bound[
                    node["backendDOMNodeId"]
                ][1]
                node["offsetrect_bound"] = backend_id_to_bound[
                    node["backendDOMNodeId"]
                ][2]

        # refine the bounding box for nodes which only appear in the accessibility tree
        node_ids = [node["nodeId"] for node in accessibility_tree]
        for refine_node_id in refine_node_ids:
            child_id = refine_node_id
            parent_idx: None | int = None
            while child_id in parent_graph:
                parent_id = parent_graph[child_id]
                parent_idx = node_ids.index(parent_id)
                child_id = parent_id
                if accessibility_tree[parent_idx]["union_bound"] is not None:
                    break

            refine_node_idx = node_ids.index(refine_node_id)

            if parent_idx is not None:
                accessibility_tree[refine_node_idx][
                    "bound"
                ] = accessibility_tree[parent_idx]["bound"]
                accessibility_tree[refine_node_idx][
                    "union_bound"
                ] = accessibility_tree[parent_idx]["union_bound"]
                accessibility_tree[refine_node_idx][
                    "offsetrect_bound"
                ] = accessibility_tree[parent_idx]["offsetrect_bound"]
            else:
                accessibility_tree[refine_node_idx]["bound"] = None
                accessibility_tree[refine_node_idx]["union_bound"] = None
                accessibility_tree[refine_node_idx]["offsetrect_bound"] = None

        return accessibility_tree

    @beartype
    def current_viewport_accessibility_tree(
        self,
        info: BrowserInfo,
        accessibility_tree: AccessibilityTree,
    ) -> AccessibilityTree:
        config = info["config"]
        subtree = []
        for node in accessibility_tree:
            if not node["union_bound"]:
                continue

            [x, y, width, height] = node["union_bound"]
            elem_left_bound = x
            elem_top_bound = y
            elem_right_bound = x + width
            elem_lower_bound = y + height

            ok = (
                elem_left_bound < config["win_right_bound"]
                and elem_right_bound >= config["win_left_bound"]
                and elem_top_bound < config["win_lower_bound"]
                and elem_lower_bound >= config["win_upper_bound"]
            )

            if ok:
                subtree.append(node)

        return subtree

    @beartype
    @staticmethod
    def parse_accessibility_tree(
        accessibility_tree: AccessibilityTree,
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "bound": node["bound"],
                        "union_bound": node["union_bound"],
                        "offsetrect_bound": node["offsetrect_bound"],
                        "text": node_str,
                    }

            except Exception as e:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree_str, obs_nodes_info

    @beartype
    @staticmethod
    def clean_accesibility_tree(tree_str: str) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                pattern = r"\[\d+\] StaticText '([^']+)'"

                match = re.search(pattern, line)
                if match:
                    static_text = match.group(1)
                    if all(
                        static_text not in prev_line
                        for prev_line in prev_lines
                    ):
                        clean_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines)

    @beartype
    def process(self, page: Page, client: CDPSession) -> str:
        # get the tab info
        open_tabs = page.context.pages
        try:
            tab_titles = [tab.title() for tab in open_tabs]
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[
                        idx
                    ] = f"Tab {idx} (current): {open_tabs[idx].title()}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {open_tabs[idx].title()}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join(
                ["Tab {idx}" for idx in range(len(open_tabs))]
            )

        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = self.fetch_browser_info(page, client)

        if self.current_viewport_only:
            self.retrieve_viewport_info(browser_info)

        if self.observation_type == "html":
            if self.current_viewport_only:
                html = self.current_viewport_html(browser_info)
                content = html
            else:
                content = page.content()
        elif self.observation_type == "":
            content = ""
        elif self.observation_type == "accessibility_tree":
            accessibility_tree = self.fetch_page_accessibility_tree(
                browser_info, client
            )
            if self.current_viewport_only:
                accessibility_tree = self.current_viewport_accessibility_tree(
                    browser_info, accessibility_tree
                )
            content, obs_nodes_info = self.parse_accessibility_tree(
                accessibility_tree
            )
            content = self.clean_accesibility_tree(content)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info
        elif self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            # Check if the current page is an image url
            if page.url.endswith((".jpg", ".jpeg", ".png")):
                print("NOTE: We are on an image page!!!")
                # Load image from current url and run captioning on it.
                if page.url not in self.url2caption and self.captioning_fn is not None:
                    try:
                        image = Image.open(
                            requests.get(page.url, stream=True).raw
                        )
                        caption = self.captioning_fn([image])[0].strip()
                        self.url2caption[page.url] = remove_unicode(caption)
                    except Exception as e:
                        print("L579 WARNING: ", e)

                content = self.url2caption.get(page.url, "Image")
            else:
                if self.captioning_fn is not None:
                    images = page.query_selector_all("img")
                    image_urls = []
                    for image in images:
                        try:
                            image_url = image.get_attribute("src")
                            if not image_url.startswith(
                                ("http://", "https://", "www.")
                            ):
                                image_url = urljoin(page.url, image_url)
                            if image_url not in self.url2caption:
                                image_urls.append(image_url)
                        except Exception as e:
                            print("L604 WARNING: ", e)

                    # Run image captioning on image_url pixels. This is for models which use captioning as a baseline.
                    if len(image_urls) > 0:
                        image_pixels = []
                        valid_urls = []
                        for url in image_urls:
                            if "data:image/svg" in url:
                                continue
                            else:
                                try:
                                    image = Image.open(
                                        requests.get(url, stream=True).raw
                                    )
                                    image_pixels.append(image)
                                    valid_urls.append(url)
                                except Exception as e:
                                    print("L616 WARNING: ", e)

                        # Caption images.
                        if image_pixels:
                            # Run in batches of 4.
                            bs = 4
                            captions = []
                            for i in range(0, len(image_pixels), bs):
                                try:
                                    captions.extend(
                                        self.captioning_fn(
                                            image_pixels[i : i + bs]
                                        )
                                    )
                                except Exception as e:
                                    print("L628 WARNING: ", e)
                                    captions.extend(
                                        [""] * len(image_pixels[i : i + bs])
                                    )
                            assert len(valid_urls) == len(
                                captions
                            ), f"len(images)={len(valid_urls)}, len(captions)={len(captions)}"
                            for image_url, caption in zip(valid_urls, captions):
                                self.url2caption[image_url] = remove_unicode(
                                    caption.strip()
                                )

                    image_idx = 0
                    for image in images:
                        try:
                            original_alt = image.get_attribute("alt") or ""
                            image_url = image.get_attribute("src")
                            if not image_url.startswith(
                                ("http://", "https://", "www.")
                            ):
                                image_url = urljoin(page.url, image_url)

                            updated_alt = original_alt

                            if image_url in self.url2caption:
                                if self.url2caption[image_url] not in updated_alt:
                                    updated_alt = f"{updated_alt}, description: {self.url2caption[image_url]}"
                            elif "data:image/svg" not in image_url:
                                print(
                                    f"WARNING: {image_url} not in self.url2caption"
                                )

                            if "url:" not in updated_alt:
                                updated_alt = f"{updated_alt}, url: {image_url}"

                            safe_updated_alt = json.dumps(updated_alt)
                            image.evaluate(
                                f"node => node.alt = {safe_updated_alt}"
                            )
                        except Exception as e:
                            print("L653 WARNING:", e)

                if (
                    self.observation_type
                    == "accessibility_tree_with_captioner"
                ):
                    accessibility_tree = self.fetch_page_accessibility_tree(
                        browser_info, client
                    )
                    if self.current_viewport_only:
                        accessibility_tree = (
                            self.current_viewport_accessibility_tree(
                                browser_info, accessibility_tree
                            )
                        )
                    content, obs_nodes_info = self.parse_accessibility_tree(
                        accessibility_tree
                    )
                    content = self.clean_accesibility_tree(content)
                    self.obs_nodes_info = obs_nodes_info
                    self.meta_data["obs_nodes_info"] = obs_nodes_info
                else:
                    content = ""  # Not used for SoM
        else:
            raise ValueError(
                f"Invalid observation type: {self.observation_type}"
            )

        self.browser_config = browser_info["config"]
        content = f"{tab_title_str}\n\n{content}"
        return content

    @beartype
    def get_element_center(self, element_id: str) -> tuple[float, float]:
        node_info = self.obs_nodes_info[element_id]
        node_bound = node_info["bound"]
        x, y, width, height = node_bound
        browser_config = self.browser_config
        b_x, b_y = (
            browser_config["win_left_bound"],
            browser_config["win_upper_bound"],
        )
        center_x = (x - b_x) + width / 2
        center_y = (y - b_y) + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )