"""
Test RAG question-answering system.
This script tests the complete RAG pipeline with sample questions.
"""

import argparse

from src.rag.embeddings import create_embedding_model
from src.rag.llm import create_llm
from src.rag.prompts import get_default_rag_prompt
from src.rag.rag_chain import create_rag_chain, query_rag
from src.rag.vector_store import create_vector_store
from src.utils.emoji_log import done, info, success, task


def test_rag_system(question: str = None):
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
    prompt = get_default_rag_prompt()

    # 5. Build RAG chain
    info("Building RAG chain...")
    rag_chain = create_rag_chain(llm=llm, vector_store=vector_store, prompt=prompt)
    done("RAG Chain built!")

    # 6. Execute query
    if question is None:
        question = "Can you provide some Seattle attractions to me?"

    info(f"Question: {question}")
    answer = query_rag(chain=rag_chain, question=question)

    # 7. Display result
    print("=" * 70)
    print("Answer:")
    print("=" * 70)
    print(answer)
    print("=" * 70)

    success("RAG test complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG system")
    parser.add_argument(
        "--question",
        help="Question to ask (default: 'Can you provide some Seattle attractions to me?')",
    )

    args = parser.parse_args()

    test_rag_system(question=args.question)
