"""
Test RAG question-answering system.
This script tests the complete RAG pipeline with sample questions.
"""

import argparse

from langchain_core.runnables import RunnableWithMessageHistory

from src.rag.cli import interactive_loop, single_query
from src.rag.embeddings import create_embedding_model
from src.rag.llm import create_llm
from src.rag.memory import get_session_history
from src.rag.prompts import get_conversational_rag_prompt
from src.rag.rag_chain import create_conversational_rag_chain_with_sources
from src.rag.vector_store import create_vector_store
from src.utils.emoji_log import done, info, success, task


def test_rag_system(
    question: str = None, session_id: str = "user_001", interactive: bool = False
):
    """
    Test the RAG system with a question.

    Args:
        question: Test question (if None, use default questions)
    """
    task("Start RAG Flow...")

    # 1. Load embedding model
    info("Loading embedding model...")
    embeddings = create_embedding_model()
    done("Embedding model loaded")

    # 2. Load vector store
    info("Loading vector store...")
    vector_store = create_vector_store(
        collection_name="travel_attractions", embeddings=embeddings
    )
    done("Vector store loaded")

    # 3. Build LLM
    info("Creating LLM...")
    llm = create_llm()
    done("LLM created")

    # 4. Get prompt
    prompt = get_conversational_rag_prompt()

    # 5. Build RAG chain
    info("Building RAG chain...")
    rag_chain = create_conversational_rag_chain_with_sources(
        llm=llm, vector_store=vector_store, prompt=prompt
    )
    done("RAG Chain built!")

    # Wrap chain with RunnableWithMessageHistory
    rag_chain_memory = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )

    # 6. Determine mode
    if interactive:
        interactive_loop(rag_chain_memory, session_id)
    else:
        single_query(rag_chain_memory, question, session_id)

    success("RAG test complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test conversational RAG system")
    parser.add_argument(
        "--question",
        help="Question to ask (default: 'Can you provide some Seattle attractions to me?')",
    )
    parser.add_argument(
        "--session",
        default="user_001",
        help="Session ID for conversation memory (default: 'user_001')",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode for multi-turn conversation",
    )

    args = parser.parse_args()

    test_rag_system(
        question=args.question, session_id=args.session, interactive=args.interactive
    )
