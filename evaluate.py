"""
evaluate.py - Evaluation harness for the insurance claim support agent.

1. Retrieval quality - Hit Rate@k and Mean Reciprocal Rank (MRR): does the
   FAISS retriever surface a chunk from the document each question is
   actually about?
2. Answer quality - an LLM-as-judge (Groq) rates each generated answer
   against a short reference answer for correctness and groundedness
   (1-5 scale).
3. Latency - end-to-end response time per question through the full agent,
   including the LangMem recall and update steps.

"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from agent import build_app, EMBEDDING_MODEL, VECTOR_STORE_DIR, GROQ_MODEL, TOP_K
from eval_dataset import EVAL_QUESTIONS

load_dotenv()

JUDGE_MODEL = GROQ_MODEL 
SLEEP_BETWEEN_QUESTIONS = 1.0


# 1. Retrieval evaluation

def evaluate_retrieval(vector_store, questions, k):
    """For each question, check whether the expected source document appears
    in the top-k retrieved chunks, and at what rank."""
    results = []
    for item in questions:
        docs = vector_store.similarity_search(item["question"], k=k)
        sources = [Path(d.metadata.get("source", "")).name for d in docs]
        expected = item["expected_source"]

        if expected in sources:
            rank = sources.index(expected) + 1
            reciprocal_rank = 1.0 / rank
        else:
            rank = None
            reciprocal_rank = 0.0

        results.append({
            "question": item["question"],
            "expected_source": expected,
            "retrieved_sources": sources,
            "hit": expected in sources,
            "rank": rank,
            "reciprocal_rank": reciprocal_rank,
        })
    return results



# 2. LLM-as-judge for answer quality


JUDGE_PROMPT = """You are evaluating an insurance support assistant's answer.

Question: {question}

Reference answer (key facts that should be present): {reference_answer}

Assistant's answer: {generated_answer}

Rate the assistant's answer on two dimensions, each on a 1-5 scale:
- correctness: does it contain the key facts from the reference answer?
- groundedness: does it stick to plausible policy details without inventing
  unrelated facts not supported by the reference?

Respond with ONLY a JSON object, no other text, in this exact format:
{{"correctness": <int 1-5>, "groundedness": <int 1-5>, "reasoning": "<one short sentence>"}}
"""


def judge_answer(judge_llm, question, reference_answer, generated_answer):
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    response = judge_llm.invoke(prompt)
    text = response.content.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
        return {
            "correctness": parsed.get("correctness"),
            "groundedness": parsed.get("groundedness"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        return {"correctness": None, "groundedness": None, "reasoning": f"Could not parse judge output: {text[:200]}"}



# 3. Full agent evaluation (answer quality + latency)


def evaluate_agent(app, judge_llm, questions):
    """Run each question through the full agent, timing it end-to-end, then
    score the answer with the LLM judge. Each question gets its own
    user_id/thread_id so the questions don't share conversation history or
    long-term memory with each other."""
    results = []
    for i, item in enumerate(questions):
        config = {"configurable": {"user_id": f"eval_user_{i}", "thread_id": f"eval_thread_{i}"}}

        start = time.perf_counter()
        result = app.invoke({"messages": [HumanMessage(content=item["question"])]}, config=config)
        elapsed = time.perf_counter() - start

        answer = result["messages"][-1].content
        judgement = judge_answer(judge_llm, item["question"], item["reference_answer"], answer)

        results.append({
            "question": item["question"],
            "generated_answer": answer,
            "latency_seconds": elapsed,
            **judgement,
        })

        time.sleep(SLEEP_BETWEEN_QUESTIONS)

    return results


# 4. Reporting

def summarize(retrieval_results, agent_results):
    n = len(retrieval_results)
    hit_rate = sum(r["hit"] for r in retrieval_results) / n
    mrr = sum(r["reciprocal_rank"] for r in retrieval_results) / n

    scored = [r for r in agent_results if r["correctness"] is not None]
    avg_correctness = sum(r["correctness"] for r in scored) / len(scored) if scored else None
    avg_groundedness = sum(r["groundedness"] for r in scored) / len(scored) if scored else None

    latencies = sorted(r["latency_seconds"] for r in agent_results)
    avg_latency = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]

    return {
        "n_questions": n,
        "retrieval_hit_rate_at_k": round(hit_rate, 3),
        "retrieval_mrr": round(mrr, 3),
        "avg_correctness_1_5": round(avg_correctness, 2) if avg_correctness is not None else None,
        "avg_groundedness_1_5": round(avg_groundedness, 2) if avg_groundedness is not None else None,
        "avg_latency_seconds": round(avg_latency, 2),
        "p50_latency_seconds": round(p50, 2),
        "p95_latency_seconds": round(p95, 2),
    }


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY in your environment or a .env file")

    print("Loading vector store and agent...\n")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    app = build_app()
    judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=0)

    print(f"=== Retrieval evaluation ({len(EVAL_QUESTIONS)} questions, k={TOP_K}) ===")
    retrieval_results = evaluate_retrieval(vector_store, EVAL_QUESTIONS, k=TOP_K)
    for r in retrieval_results:
        status = "HIT " if r["hit"] else "MISS"
        rank_str = f"rank {r['rank']}" if r["rank"] else "not retrieved"
        print(f"  [{status}] ({rank_str}) {r['question'][:65]}")

    print(f"\n=== Agent evaluation: answer quality + latency ({len(EVAL_QUESTIONS)} questions) ===")
    print("(this calls Groq twice per question - once to answer, once to judge)\n")
    agent_results = evaluate_agent(app, judge_llm, EVAL_QUESTIONS)
    for r in agent_results:
        print(
            f"  correctness={r['correctness']}  groundedness={r['groundedness']}  "
            f"latency={r['latency_seconds']:.2f}s  {r['question'][:50]}"
        )

    summary = summarize(retrieval_results, agent_results)

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    with open("eval_results.json", "w") as f:
        json.dump(
            {"summary": summary, "retrieval_results": retrieval_results, "agent_results": agent_results},
            f,
            indent=2,
        )
    print("\nSaved detailed results to eval_results.json")


if __name__ == "__main__":
    main()