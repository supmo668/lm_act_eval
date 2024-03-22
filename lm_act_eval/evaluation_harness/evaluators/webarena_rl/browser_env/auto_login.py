"""Script to automatically login each website"""
import argparse
import glob
import os
import time
import yaml

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from playwright.sync_api import sync_playwright

from .env_config import config as browse_config

def is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True
) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True

    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(
        headless=browse_config.browse.HEADLESS, slow_mo=browse_config.browse.SLOW_MO)
    context = browser.new_context(storage_state=storage_state)
    page = context.new_page()
    page.goto(url)
    time.sleep(1)
    d_url = page.url
    content = page.content()
    context_manager.__exit__()
    if keyword:
        return keyword not in content
    else:
        if url_exact:
            return d_url != url
        else:
            return url not in d_url

def page_handler1(page, url_endpoint, slug):
    site_config = getattr(browse_config, 'slug')
    page.goto(url_endpoint, timeout=0)
    page.get_by_label("Email", exact=True).fill(site_config.username)
    page.get_by_label("Password", exact=True).fill(site_config.password)
    page.get_by_role("button", name="Sign In").click()

def page_handler2(page, url_endpoint, slug):
    site_config = getattr(browse_config, 'slug')
    page.goto(url_endpoint, timeout=0)
    page.locator("#email").fill(site_config.username)
    page.locator("#password").fill(site_config.password)
    page.get_by_role("button", name="Log in").click()

def renew_comb(comb: list[str]) -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=browse_config.browse.HEADLESS, slow_mo=browse_config.browse.SLOW_MO)
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(0)
    for site_name in comb:
        match site_name:
            case "shopping":
                page_handler1(
                    page, "{token}/customer/account/login/", site_name)
            case "reddit": 
                page_handler1(
                    page, "{token}/login", site_name)

            case "classifieds":
                page_handler2()
        

    context.storage_state(path=f"./.auth/{'.'.join(comb)}_state.json")

    context_manager.__exit__()


def main() -> None:
    for site in list(browse_config.params.sites.keys()):
        renew_comb([site])

    for c_file in glob.glob("./.auth/*.json"):
        comb = c_file.split("/")[-1].rsplit("_", 1)[0].split(".")
        for cur_site in comb:
            site_config = getattr(browse_config, cur_site)
            assert not is_expired(Path(c_file), site_config.url, site_config.keyword, site_config.match), site_config.url


if __name__ == "__main__":
    main()
