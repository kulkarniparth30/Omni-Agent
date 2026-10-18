"""
evals/eval_ops.py

Operational Evaluation: Latency and Reliability Metrics.
Metrics:
  - latency.e2e_p95_ms
  - latency.e2e_p50_ms
  - reliability.success_rate
  - reliability.error_rate
"""

from typing import List, Dict, Any
import numpy as np

def eval_ops_metrics(latencies_ms: List[float], errors_count: int, total_turns: int) -> Dict[str, float]:
    """
    Calculate operational latency percentiles and reliability rates.
    """
    if not latencies_ms:
        return {
            "latency.e2e_p95_ms": 0.0,
            "latency.e2e_p50_ms": 0.0,
            "reliability.success_rate": 1.0,
            "reliability.error_rate": 0.0
        }

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    success_rate = (total_turns - errors_count) / max(1, total_turns)
    error_rate = errors_count / max(1, total_turns)

    return {
        "latency.e2e_p95_ms": round(p95, 2),
        "latency.e2e_p50_ms": round(p50, 2),
        "reliability.success_rate": round(success_rate, 4),
        "reliability.error_rate": round(error_rate, 4)
    }
