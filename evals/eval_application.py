"""
evals/eval_application.py

Application-Level Quality Evaluation (G-Eval style).
Metrics:
  - tool_selection.accuracy
  - answer_correctness.score
"""

from typing import List, Dict, Any

def eval_application_quality(all_turns: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate overall application quality and tool routing precision.
    """
    tool_matches = 0
    correctness_scores = []

    for turn in all_turns:
        expected_tools = sorted(turn.get("expected_tools", []))
        actual_tools = sorted(turn.get("tools_used", []))
        response = turn.get("response", "").lower()
        expected_contains = turn.get("expected_contains", [])

        # 1. Tool Selection Precision
        # Did the agent invoke the expected tool? (Or cleanly invoke none if none expected)
        if expected_tools == actual_tools:
            tool_matches += 1
        elif expected_tools and any(t in actual_tools for t in expected_tools):
            # Partial hit
            tool_matches += 0.8
        elif not expected_tools and not actual_tools:
            tool_matches += 1

        # 2. Answer Correctness
        if expected_contains:
            hits = sum(1 for kw in expected_contains if kw.lower() in response)
            correctness = hits / len(expected_contains)
            correctness_scores.append(correctness)
        else:
            correctness_scores.append(1.0 if len(response) > 5 else 0.5)

    tool_accuracy = tool_matches / len(all_turns) if all_turns else 1.0
    avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 1.0

    return {
        "tool_selection.accuracy": round(tool_accuracy, 4),
        "answer_correctness.score": round(avg_correctness, 4)
    }
