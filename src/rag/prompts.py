"""
Prompt template management functions.
This module provides pure functions for creating and managing prompt templates
for RAG question-answering. All functions are stateless and composable.
"""

from langchain_core.prompts import PromptTemplate


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
