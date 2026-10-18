"""
evals/eval_retriever.py

Component-Level Evaluation for the FAISS Retriever.
Metrics:
  - contextual_recall
  - contextual_precision
"""

from typing import List, Dict, Any
from app.services.pdf_store import get_retriever

def eval_retriever_for_thread(thread_id: str, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate FAISS retriever performance for a set of golden RAG questions.
    Returns:
      {
        "contextual_recall.score": float,
        "contextual_precision.score": float
      }
    """
    retriever = get_retriever(thread_id)
    if not retriever:
        return {
            "contextual_recall.score": 0.0,
            "contextual_precision.score": 0.0,
        }

    recall_scores = []
    precision_scores = []

    for tc in test_cases:
        query = tc["input"]
        expected_keywords = tc.get("expected_contains", [])
        if not expected_keywords:
            continue

        results = retriever.invoke(query)
        if not results:
            recall_scores.append(0.0)
            precision_scores.append(0.0)
            continue

        # Check recall: how many expected keywords appear in retrieved chunks
        all_content = " ".join([d.page_content.lower() for d in results])
        hits = sum(1 for kw in expected_keywords if kw.lower() in all_content)
        recall = hits / len(expected_keywords) if expected_keywords else 1.0
        recall_scores.append(recall)

        # Check precision: is the keyword present in top chunk (rank 1)?
        top_chunk_text = results[0].page_content.lower()
        top_hits = sum(1 for kw in expected_keywords if kw.lower() in top_chunk_text)
        precision = top_hits / len(expected_keywords) if expected_keywords else 1.0
        precision_scores.append(precision)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 1.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 1.0

    return {
        "contextual_recall.score": round(avg_recall, 4),
        "contextual_precision.score": round(avg_precision, 4)
    }
