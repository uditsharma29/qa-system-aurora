import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Path to the local data file
LOCAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "messages.json")

def get_messages() -> List[dict]:
    """
    Fetches member messages from a local JSON file instead of a live API.
    """
    logger.info(f"Loading messages from local file: {LOCAL_DATA_PATH}")
    try:
        with open(LOCAL_DATA_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
            items = payload.get("items", [])

        logger.info(f"Loaded {len(items)} messages from the local file.")
        return items
    except FileNotFoundError:
        logger.error(f"Data file not found at: {LOCAL_DATA_PATH}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error reading or parsing data file: {e}")
        return []
