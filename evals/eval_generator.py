"""
evals/eval_generator.py

Component-Level Evaluation for the Generator.
Metrics:
  - faithfulness (groundedness in context)
  - answer_relevancy (responsiveness to query)
"""

from typing import List, Dict, Any

def eval_generator_responses(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate faithfulness and answer relevancy across execution records.
    `records` contains entries with: {"input", "response", "expected_contains", "context"}
    """
    faithfulness_scores = []
    relevancy_scores = []

    for item in records:
        response = item.get("response", "").strip()
        expected = item.get("expected_contains", [])
        context = " ".join(item.get("context", []))

        if not response:
            faithfulness_scores.append(0.0)
            relevancy_scores.append(0.0)
            continue

        # 1. Faithfulness: If context is available, does the answer stay grounded?
        if context:
            # Check overlap between response sentences/words and context
            resp_words = set(response.lower().split())
            ctx_words = set(context.lower().split())
            overlap = len(resp_words.intersection(ctx_words))
            faith = min(1.0, (overlap / max(1, len(resp_words))) * 1.5)
            faithfulness_scores.append(faith)
        else:
            # Non-RAG turn: default grounded
            faithfulness_scores.append(1.0)

        # 2. Answer Relevancy: Did the response contain the expected golden elements?
        if expected:
            hits = sum(1 for kw in expected if kw.lower() in response.lower())
            rel = hits / len(expected)
            relevancy_scores.append(rel)
        else:
            relevancy_scores.append(1.0 if len(response) > 5 else 0.5)

    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 1.0
    avg_rel = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 1.0

    return {
        "faithfulness.score": round(avg_faith, 4),
        "answer_relevancy.score": round(avg_rel, 4)
    }
