"""
agent.py - LangGraph agent for insurance claim Q&A (RAG + LangMem)

Graph:
    recall_memory -> retrieve -> generate -> update_memory

- recall_memory: looks up any relevant long-term memories LangMem has stored
  about this user.
- retrieve: embeds the user's question and pulls the top matching chunks
  from the FAISS index built by ingest.py.
- generate: calls Groq with the retrieved chunks + remembered context +
  conversation history, and writes the answer back to messages.
- update_memory: hands the full conversation to LangMem's memory manager,
  which extracts/updates durable facts in the store.
"""

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.config import get_store, get_config
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384  # all-MiniLM-L6-v2 output size
VECTOR_STORE_DIR = "vector_store"
TOP_K = 3

MEMORY_NAMESPACE = ("insurance_agent", "{user_id}")
MEMORY_INSTRUCTIONS = (
    "Extract durable facts about this user's insurance situation - for "
    "example, which policy type (auto, home, health) they're asking about, "
    "details of a claim they've mentioned, or their name. Keep each memory "
    "short (one sentence). Don't store the assistant's answers, only facts "
    "about the user and their situation."
)

# State

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: str
    memory_context: str

# Graph construction


def build_graph(vector_store, llm, memory_manager, store, checkpointer):
    """Wire up the four nodes into a compiled LangGraph app.

    vector_store, llm, and memory_manager are passed in (rather than created
    here) so the same graph-building code can be reused with real
    components in main() and with lightweight fakes in tests.
    """

    def recall_memory(state: AgentState) -> dict:
        config = get_config()
        store = get_store()
        user_id = config["configurable"].get("user_id", "default")
        namespace = ("insurance_agent", user_id)

        question = state["messages"][-1].content
        items = store.search(namespace, query=question, limit=3)
        memories = [item.value["content"]["content"] for item in items]
        return {"memory_context": "\n".join(f"- {m}" for m in memories)}

    def retrieve(state: AgentState) -> dict:
        question = state["messages"][-1].content
        docs = vector_store.similarity_search(question, k=TOP_K)
        context_parts = [
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
        return {"retrieved_context": "\n\n---\n\n".join(context_parts)}

    def generate(state: AgentState) -> dict:
        system_prompt = (
            "You are an insurance claim support assistant for Harborlight "
            "Insurance. Answer the user's question using ONLY the policy "
            "information below. If the answer isn't there, say you don't "
            "have that information and suggest contacting Harborlight "
            "support at 1-800-555-0142. Briefly mention which document your "
            "answer is based on.\n\n"
            f"Relevant policy information:\n{state['retrieved_context']}\n\n"
            f"What you remember about this user:\n"
            f"{state['memory_context'] or 'Nothing yet.'}"
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def update_memory(state: AgentState) -> dict:
        config = get_config()
        memory_manager.invoke({"messages": state["messages"]}, config=config)
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("recall_memory", recall_memory)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("update_memory", update_memory)

    graph.add_edge(START, "recall_memory")
    graph.add_edge("recall_memory", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile(store=store, checkpointer=checkpointer)


# CLI entry point

def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY in your environment or a .env file")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    store = InMemoryStore(index={"dims": EMBEDDING_DIMS, "embed": embeddings})
    memory_manager = create_memory_store_manager(
        llm,
        namespace=MEMORY_NAMESPACE,
        instructions=MEMORY_INSTRUCTIONS,
        enable_inserts=True,
        query_limit=3,
        store=store,
    )

    checkpointer = InMemorySaver()
    app = build_graph(vector_store, llm, memory_manager, store, checkpointer)

    # user_id scopes long-term memory, thread_id scopes conversation history.
    # Both are in-memory here, so they reset when the script exits.
    config = {"configurable": {"user_id": "demo_user", "thread_id": "demo_thread"}}

    print("Insurance Claim Support Agent (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = app.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()