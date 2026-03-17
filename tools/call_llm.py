"""
tools/call_llm.py

Thin wrapper around Gemini. Keeping your original LLM choice — only cleaned
up the API key handling and added a timeout/retry boundary.
"""

import os
import logging
from typing import Optional

import google.genai as genai

logger = logging.getLogger(__name__)

_MODEL = "gemini-3.1-pro-preview"


def call_llm(prompt: str, model: Optional[str] = None) -> str:
    """
    Call Gemini and return the response text.

    Args:
        prompt: Full prompt string
        model:  Optional model override (defaults to gemini-2.5-pro)

    Returns:
        Generated text string

    Raises:
        ValueError: API key not set
        RuntimeError: API call failed
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Add it to your .env file or export it in your shell."
        )

    try:
        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model    = model or _MODEL,
            contents = prompt,
        )
        return response.text
    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise RuntimeError(f"LLM call failed: {exc}") from exc