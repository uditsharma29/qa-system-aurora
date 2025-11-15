import logging
import os
from typing import List, Tuple

import requests

logger = logging.getLogger(__name__)

MESSAGES_API_URL = "https://november7-730026606190.europe-west1.run.app/messages"
MESSAGES_API_PAGE_SIZE = 1000
MESSAGES_API_TIMEOUT = 20


def _fetch_page(offset: int) -> Tuple[List[dict], int]:
    """Fetch a single page from the API."""
    params = {"limit": MESSAGES_API_PAGE_SIZE, "skip": offset}
    response = requests.get(
        MESSAGES_API_URL,
        params=params,
        timeout=MESSAGES_API_TIMEOUT,
        allow_redirects=True,
    )
    response.raise_for_status()

    payload = response.json()
    items = payload.get("items")
    total = payload.get("total")

    if not isinstance(items, list):
        raise ValueError("Unexpected API response shape: 'items' missing or invalid.")

    return items, total


def get_messages() -> List[dict]:
    """
    Fetches member messages from the public API, paging through the dataset until the
    service returns an empty page (or the reported total is reached), and returns the
    raw message objects from the API (including user metadata).
    """
    records: List[dict] = []
    offset = 0

    page = 0
    try:
        while True:
            page += 1
            try:
                items, reported_total = _fetch_page(offset)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in {400, 401, 402, 403, 404, 405}:
                    logger.warning(
                        "Received %s from API while paging; returning records fetched so far.",
                        status,
                    )
                    break
                raise

            records.extend(items)

            logger.info(
                "Fetched page %s (offset=%s, page_size=%s, cumulative_messages=%s, total=%s)",
                page,
                offset,
                len(items),
                len(records),
                reported_total,
            )

            if not items or len(records) >= reported_total:
                break

            offset += len(items) or MESSAGES_API_PAGE_SIZE

    except requests.exceptions.RequestException as e:
        logger.error("Request to messages API failed (%s).", e)
        raise

    logger.info(
        "Fetched %s messages from the public API.",
        len(records),
    )
    return records
