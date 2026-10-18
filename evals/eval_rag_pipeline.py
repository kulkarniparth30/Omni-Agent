"""
evals/eval_rag_pipeline.py

Pipeline-Level Evaluation: The Complete RAG Triad.
Measures the three legs of the RAG triangle:
  1. Contextual Relevancy (Query <-> Context)
  2. Faithfulness (Context -> Response)
  3. Answer Relevancy (Query <-> Response)
"""

from typing import List, Dict, Any

def eval_rag_triad(rag_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute RAG Triad metrics across RAG execution records.
    Returns:
      {
        "contextual_relevancy.score": float,
        "faithfulness.score": float,
        "answer_relevancy.score": float,
        "rag_triad.score": float
      }
    """
    if not rag_records:
        return {
            "contextual_relevancy.score": 1.0,
            "faithfulness.score": 1.0,
            "answer_relevancy.score": 1.0,
            "rag_triad.score": 1.0
        }

    c_rel_list = []
    faith_list = []
    a_rel_list = []

    for r in rag_records:
        query = r.get("input", "").lower()
        context_list = r.get("context", [])
        context_str = " ".join(context_list).lower()
        response = r.get("response", "").lower()
        expected = r.get("expected_contains", [])

        # 1. Contextual Relevancy: Query terms in Context
        q_words = set(query.split()) - {"what", "who", "when", "where", "how", "the", "is", "of", "in", "and", "to"}
        if q_words:
            c_hits = sum(1 for w in q_words if w in context_str)
            c_rel_list.append(min(1.0, c_hits / len(q_words)))
        else:
            c_rel_list.append(1.0)

        # 2. Faithfulness: Response grounded in Context
        if context_str and response:
            r_words = set(response.split())
            ctx_words = set(context_str.split())
            overlap = len(r_words.intersection(ctx_words))
            faith = min(1.0, (overlap / max(1, len(r_words))) * 1.5)
            faith_list.append(faith)
        else:
            faith_list.append(1.0)

        # 3. Answer Relevancy: Expected keywords in Response
        if expected:
            e_hits = sum(1 for kw in expected if kw.lower() in response)
            a_rel_list.append(e_hits / len(expected))
        else:
            a_rel_list.append(1.0)

    avg_c_rel = sum(c_rel_list) / len(c_rel_list) if c_rel_list else 1.0
    avg_faith = sum(faith_list) / len(faith_list) if faith_list else 1.0
    avg_a_rel = sum(a_rel_list) / len(a_rel_list) if a_rel_list else 1.0

    # Harmonic mean of the 3 triad scores
    denom = (1.0 / max(0.01, avg_c_rel)) + (1.0 / max(0.01, avg_faith)) + (1.0 / max(0.01, avg_a_rel))
    triad_harmonic = 3.0 / denom

    return {
        "contextual_relevancy.score": round(avg_c_rel, 4),
        "faithfulness.score": round(avg_faith, 4),
        "answer_relevancy.score": round(avg_a_rel, 4),
        "rag_triad.score": round(triad_harmonic, 4)
    }
