"""
LLM management functions.

This module provides pure functions for creating and testing Google Gemini LLM.
All functions are stateless and composable.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.emoji_log import error

load_dotenv()


# =======================================
# 1. Create LLM
# =======================================
def create_llm(
    model: str = "gemini-2.5-flash",
    api_key: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> ChatGoogleGenerativeAI:
    """
    Create a Google Gemini LLM instance.

    Args:
        model: Gemini model name
        temperature: Controls randomness (0-1)
        max_tokens: Maximum response length
        api_key: Google API key (optional, reads from env)

    Returns:
        ChatGoogleGenerativeAI instance
    """

    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        error("API key is required!")
        raise ValueError("GOOGLE_API_KEY is required")

    return ChatGoogleGenerativeAI(
        model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens
    )


# =======================================
# 2. Test LLM Connection
# =======================================
def test_llm_connection(llm: ChatGoogleGenerativeAI, query: str) -> str:
    """
    Test LLM connection with a simple query.

    Args:
        llm: ChatGoogleGenerativeAI instance
        query: Test query string

    Returns:
        LLM response content as string

    Example:
        >>> llm = create_llm()
        >>> response = test_llm_connection(llm, "Hello!")
        >>> print(response)
        'Hello! How can I help you today?'
    """
    response = llm.invoke(query)
    return response.content
