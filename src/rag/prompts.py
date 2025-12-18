"""
Prompt template management functions.
This module provides pure functions for creating and managing prompt templates
for RAG question-answering. All functions are stateless and composable.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


# =======================================
# 1. Default RAG Prompt
# =======================================
def get_default_rag_prompt() -> PromptTemplate:
    """
    Get the default RAG prompt template for travel question-answering.

    Returns:
        PromptTemplate instance configured for travel Q&A

    Example:
        >>> prompt = get_default_rag_prompt()
        >>> formatted = prompt.format(context="...", question="...")
    """
    template = """You are a helpful travel assistant specializing in tourist attractions.
Use the following context to answer the question. The context contains information about various tourist attractions including their names, locations, and descriptions.
Context:
{context}
Question: {question}
Instructions:
- Answer based ONLY on the information provided in the context above
- If the context doesn't contain relevant information, say "I don't have information about that in my database"
- Be concise and helpful
- Include specific attraction names and locations when relevant
Answer:"""

    prompt = PromptTemplate.from_template(template)

    return prompt


# =======================================
# 2. Custom RAG Prompt
# =======================================
def create_custom_rag_prompt(system_role: str, instructions: str) -> PromptTemplate:
    """
    Create a custom RAG prompt template.

    Args:
        system_role: Description of the assistant's role
        instructions: Specific instructions for answering

    Returns:
        PromptTemplate instance with custom configuration

    Example:
        >>> prompt = create_custom_rag_prompt(
        ...     system_role="You are a museum expert",
        ...     instructions="Focus on historical details"
        ... )
    """
    template = f"""You are a helpful {system_role}.
Use the following context to answer the question. The context contains information about various tourist attractions including their names, locations, and descriptions.

Context:
{{context}}

Question: {{question}}

Instructions:
{instructions}
Answer:"""

    prompt = PromptTemplate.from_template(template)

    return prompt


def get_conversational_rag_prompt() -> ChatPromptTemplate:
    """
    Get conversational RAG prompt template with chat history support.

    This prompt template includes:
    - System message with context placeholder
    - MessagesPlaceholder for chat history
    - Human message with question placeholder

    Returns:
        ChatPromptTemplate for conversational RAG with MessagesPlaceholder

    Example:
        >>> from src.rag.prompts import get_conversational_rag_prompt
        >>> from src.rag.memory import get_session_history
        >>> from langchain_core.runnables.history import RunnableWithMessageHistory
        >>>
        >>> prompt = get_conversational_rag_prompt()
        >>> # Use with RunnableWithMessageHistory for conversation memory

    Note:
        This prompt expects three variables:
        - context: Retrieved document context (string)
        - chat_history: Previous conversation messages (list)
        - question: Current user question (string)
    """
    conversational_template = [
        # This is a tuple, tell AI the system message
        (
            "system",
            """You are a helpful travel assistant specializing in tourist attractions.
Use the following context to answer the question. The context contains information about various tourist attractions.
Context:
{context}
Instructions:
- Answer based ONLY on the information provided in the context above
- If the context doesn't contain relevant information, say "I don't have information about that in my database"
- Be concise and helpful
- Use the conversation history to understand context and pronouns""",
        ),
        # This is a placeholder here to insert the chat history
        MessagesPlaceholder(variable_name="chat_history"),
        # Another tuple, user message
        ("human", "{question}"),
    ]

    return ChatPromptTemplate.from_messages(conversational_template)
