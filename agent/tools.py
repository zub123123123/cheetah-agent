import logging
import time

import requests

from .config import WIKI_USER_AGENT

logger = logging.getLogger(__name__)

WIKI_API_URL = "https://{lang}.wikipedia.org/w/api.php"
REQUEST_TIMEOUT = 10


def _headers() -> dict:
    return {"User-Agent": WIKI_USER_AGENT}


def search_wikipedia(query: str, lang: str = "en", limit: int = 3) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    logger.debug("search_wikipedia: query=%r lang=%s", query, lang)
    start = time.perf_counter()
    response = requests.get(
        WIKI_API_URL.format(lang=lang),
        params=params,
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json().get("query", {}).get("search", [])
    titles = [item["title"] for item in results]
    logger.info(
        "search_wikipedia: query=%r lang=%s -> %d results in %.2fs",
        query,
        lang,
        len(titles),
        time.perf_counter() - start,
    )
    return titles


def fetch_article_text(title: str, lang: str = "en", max_chars: int = 12000) -> str | None:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    logger.debug("fetch_article_text: title=%r lang=%s", title, lang)
    start = time.perf_counter()
    response = requests.get(
        WIKI_API_URL.format(lang=lang),
        params=params,
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    elapsed = time.perf_counter() - start
    if not page or "missing" in page:
        logger.info("fetch_article_text: title=%r lang=%s -> missing (%.2fs)", title, lang, elapsed)
        return None
    text = page.get("extract", "")
    result = text[:max_chars] if text else None
    logger.info(
        "fetch_article_text: title=%r lang=%s -> %d chars (%.2fs)",
        title,
        lang,
        len(result) if result else 0,
        elapsed,
    )
    return result
