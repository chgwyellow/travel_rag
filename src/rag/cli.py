"""
Command-line interface for RAG system.
Provides single query and interactive modes.
"""

from langchain_core.runnables import Runnable

from src.rag.citations import format_citations_detailed
from src.rag.memory import clear_session_history, get_session_history
from src.utils.emoji_log import error, info


def single_query(
    rag_chain_memory: Runnable,
    question: str,
    session_id: str,
    show_sources: bool = True,
) -> dict:
    """
    Execute a single query and display results.

    Args:
        rag_chain_memory: RAG chain with memory
        question: Question to ask
        session_id: Session ID for conversation memory
        show_sources: Whether to display source documents

    Returns:
        Result dictionary with answer and source_documents
    """
    if question is None:
        question = "Can you provide some Seattle attractions to me?"

    result = rag_chain_memory.invoke(
        input={"question": question},
        config={"configurable": {"session_id": session_id}},
    )

    # Display result
    print("=" * 70)
    print("Answer:")
    print(result["answer"])
    print()

    # Display source documents
    if show_sources:
        print("=" * 70)
        print("SOURCE DOCUMENTS:")
        print(format_citations_detailed(result["source_documents"]))

    return result


def interactive_loop(rag_chain_memory: Runnable, session_id: str) -> None:
    """
    Interactive mode for multi-turn conversation.

    Args:
        rag_chain_memory: RAG chain with memory
        session_id: Session ID for conversation memory
    """
    print("=" * 70)
    print("Interactive RAG Mode")
    print("Commands:")
    print("- Type your question to ask")
    print("- 'quit' or 'exit' to quit")
    print("- 'clear' to clear conversation history")
    print("- 'history' to show conversation history")
    print("=" * 70)
    print()

    turn = 0

    while True:
        turn += 1
        question = input(f"Turn {turn} Your Question: ").strip()

        # quit or exit keywords
        if question.lower() in ["quit", "exit"]:
            info("Exiting interactive mode...")
            break

        # clear keyword
        if question.lower() == "clear":
            clear_session_history(session_id=session_id)
            info("Conversation history cleared!")
            turn = 0
            continue

        # history keyword
        if question.lower() == "history":
            history = get_session_history(session_id=session_id)

            for i, msg in enumerate(history.messages, 1):
                role = (
                    "User"
                    if msg.__class__.__name__ == "HumanMessage"
                    else "AI Assistant"
                )
                content = (
                    msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                )

                print(f"{i}. {role}: {content}")

            turn -= 1  # show history is not a conversation turn
            continue

        # Empty question
        if not question:
            turn -= 1  # not a conversation turn
            continue

        try:
            result = rag_chain_memory.invoke(
                input={"question": question},
                config={"configurable": {"session_id": session_id}},
            )
            # Display result
            print("=" * 70)
            print("Answer:")
            print(result["answer"])
            print()

            # Display compact sources
            if result["source_documents"]:
                print("=" * 70)
                print(f"SOURCES ({len(result['source_documents'])} documents):")
                print("=" * 70)
                for i, doc in enumerate(result["source_documents"][:3], 1):
                    name = doc.metadata.get("name", "Unknown")
                    city = doc.metadata.get("city", "Unknown")
                    print(f"  [{i}] {name} ({city})")
        except Exception as e:
            error(f"Error: {e}")
            turn -= 1
