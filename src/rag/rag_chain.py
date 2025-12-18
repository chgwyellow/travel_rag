"""
RAG chain management functions.
This module provides pure functions for creating and managing RAG chains.
All functions are stateless and composable.
"""

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
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


# =======================================
# 4. Create conversational rag chain with sources
# =======================================
def create_conversational_rag_chain_with_sources(
    llm: ChatGoogleGenerativeAI,
    vector_store: Chroma,
    prompt: ChatPromptTemplate,
    k: int = 5,
) -> Runnable:
    """
    Create a conversational RAG chain that returns both answer and source documents.
    This chain:
    1. Preserves chat history using RunnablePassthrough.assign()
    2. Retrieves relevant documents from vector store
    3. Generates answer using LLM with conversation context
    4. Returns both answer and source documents using RunnableParallel
    Args:
        llm: ChatGoogleGenerativeAI instance
        vector_store: Chroma vector store instance
        prompt: ChatPromptTemplate with MessagesPlaceholder for chat_history
        k: Number of documents to retrieve (default: 5)

    Returns:
        Runnable chain that returns {"answer": str, "source_documents": list[Document]}

    Example:
        >>> from src.rag.prompts import get_conversational_rag_prompt
        >>> from src.rag.llm import create_llm
        >>> from src.rag.vector_store import create_vector_store
        >>> from src.rag.embeddings import create_embedding_model
        >>>
        >>> # Setup
        >>> embeddings = create_embedding_model()
        >>> vector_store = create_vector_store("travel_attractions", embeddings)
        >>> llm = create_llm()
        >>> prompt = get_conversational_rag_prompt()
        >>>
        >>> # Create chain
        >>> chain = create_conversational_rag_chain_with_sources(
        ...     llm, vector_store, prompt
        ... )
        >>>
        >>> # Use with RunnableWithMessageHistory
        >>> from langchain_core.runnables.history import RunnableWithMessageHistory
        >>> from src.rag.memory import get_session_history
        >>>
        >>> conversational_chain = RunnableWithMessageHistory(
        ...     chain,
        ...     get_session_history,
        ...     input_messages_key="question",
        ...     history_messages_key="chat_history",
        ...     output_messages_key="answer"
        ... )
    Note:
        This chain must be wrapped with RunnableWithMessageHistory to enable
        conversation memory. The output_messages_key="answer" is required to
        tell RunnableWithMessageHistory where to find the answer in the output dict.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    # Focus on retrieve the final answer for user
    answer_chain = prompt | llm | StrOutputParser()

    chain = RunnablePassthrough.assign(
        # Create docs, context, question
        docs=lambda x: retriever.invoke(x["question"]),
        context=lambda x: format_docs(retriever.invoke(x["question"])),
    ) | RunnableParallel(
        {"answer": answer_chain, "source_documents": lambda x: x["docs"]}
    )

    return chain
