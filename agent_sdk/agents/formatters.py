"""Shared response post-processing utilities for all agent services."""

import logging
import re

logger = logging.getLogger(__name__)


def fix_flash_card_format(text: str) -> str:
    """Post-process flash card responses to enforce consistent ### heading format.

    LLMs occasionally use ## or #### instead of ### for card headings despite
    explicit instructions. This fixes it and strips any preamble before the first card.
    """
    # Normalize ## headers to ### (but not #### — only fix exact ## prefix)
    text = re.sub(r'^## (?!#)', '### ', text, flags=re.MULTILINE)
    # Normalize #### headers to ### (too deep)
    text = re.sub(r'^#### ', '### ', text, flags=re.MULTILINE)

    # Strip any preamble text before the first ### card heading
    first_card = re.search(r'^### ', text, re.MULTILINE)
    if first_card:
        text = text[first_card.start():]

    card_count = len(re.findall(r'^### ', text, re.MULTILINE))
    if card_count < 3:
        logger.warning("Flash card response has only %d cards", card_count)

    return text


# Backward-compatible alias used in agent app.py imports
_fix_flash_card_format = fix_flash_card_format
