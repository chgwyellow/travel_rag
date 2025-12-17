"""
RAG chain management functions.
This module provides pure functions for creating and managing RAG chains.
All functions are stateless and composable.
"""

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


# =======================================
# 1. Format Documents
# =======================================
def format_docs(docs) -> str:
    """
    Format retrieved documents into a single string.

    Args:
        docs: List of Document objects from retriever

    Returns:
        Formatted string with all document contents

    Example:
        >>> docs = retriever.invoke("query")
        >>> context = format_docs(docs)
        >>> print(context)
        'Name: Space Needle\n\nName: Pike Place...'
    """

    return "\n\n".join(doc.page_content for doc in docs)


# =======================================
# 2. Create RAG Chain
# =======================================
def create_rag_chain(
    llm: ChatGoogleGenerativeAI,
    vector_store: Chroma,
    prompt: PromptTemplate,
    k: int = 5,
):
    """
    Create a complete RAG chain.

    Args:
        llm: ChatGoogleGenerativeAI instance
        vector_store: Chroma vector store instance
        prompt: PromptTemplate for formatting
        k: Number of documents to retrieve (default: 5)

    Returns:
        Runnable RAG chain

    Example:
        >>> chain = create_rag_chain(llm, vector_store, prompt)
        >>> answer = chain.invoke("What is the Space Needle?")
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =======================================
# 3. Query RAG Chain
# =======================================
def query_rag(chain, question: str) -> str:
    """
    Query the RAG chain with a question.

    Args:
        chain: RAG chain from create_rag_chain()
        question: User's question

    Returns:
        Answer string from LLM

    Example:
        >>> answer = query_rag(chain, "What are popular attractions?")
        >>> print(answer)
        'Some popular attractions include...'
    """
    return chain.invoke(question)
