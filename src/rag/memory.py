"""
Conversation memory management for RAG system.

This module provides functions to manage chat message histories for different sessions.
Each session maintains its own isolated conversation history, allowing multiple users
to have independent conversations with the RAG system.

Key Functions:
    - get_session_history(): Get or create chat history for a session
    - clear_session_history(): Clear history for a specific session
    - get_all_sessions(): List all active sessions
    - clear_all_sessions(): Clear all session histories

Example:
    >>> from src.rag.memory import get_session_history
    >>>
    >>> # Get history for a user
    >>> history = get_session_history("user_123")
    >>> history.add_user_message("What is the Space Needle?")
    >>> history.add_ai_message("The Space Needle is...")
    >>>
    >>> # Check messages
    >>> print(len(history.messages))  # 2
    >>>
    >>> # Clear when done
    >>> clear_session_history("user_123")

Note:
    This module uses a simple in-memory dictionary to store sessions.
    For production use, consider using a persistent storage backend
    (e.g., Redis, PostgreSQL) for session management.
"""

from langchain_community.chat_message_histories import ChatMessageHistory

from src.utils.emoji_log import warn

# Save all chat history
_session_store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Get or create chat message history for a session.

    Args:
        session_id: Unique identifier for the conversation session

    Returns:
        ChatMessageHistory instance for the session
    """

    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()

    return _session_store[session_id]


def clear_session_history(session_id: str) -> None:
    """
    Clear chat history for a specific session.

    Args:
        session_id: Session to clear
    """
    if session_id in _session_store:
        del _session_store[session_id]
        warn(f"{session_id} session has been deleted!")
    else:
        warn(f"{session_id} session not found!")


def get_all_sessions() -> list[str]:
    """
    Get list of all active session IDs.

    Returns:
        List of session IDs
    """
    return list(_session_store.keys())


def clear_all_sessions() -> None:
    """
    Clear all session histories.
    """
    _session_store.clear()

    warn("All sessions have been deleted!")
