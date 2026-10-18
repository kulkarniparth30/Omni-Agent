"""
evals/run_suite.py

Orchestrator for the full 3-tier evaluation suite (Quality + Safety + Operational).
Flattens all metrics into dotted IDs, stamps run metadata, writes a snapshot JSON,
and prints an executive scorecard.

Usage:
    python -m evals.run_suite               # Writes to evals/baselines/candidate.json
    python -m evals.run_suite --baseline    # Writes to evals/baselines/baseline.json
    python -m evals.run_suite --category rag # Run only RAG evaluations
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import httpx
from dotenv import load_dotenv

from evals.harness import load_goldens, ensure_fixture_pdf_uploaded
from evals.metric_registry import rule_for, METRIC_REGISTRY
from evals.eval_retriever import eval_retriever_for_thread
from evals.eval_generator import eval_generator_responses
from evals.eval_rag_pipeline import eval_rag_triad
from evals.eval_application import eval_application_quality
from evals.eval_safety import eval_safety_gates
from evals.eval_ops import eval_ops_metrics

load_dotenv()

BASELINE_PATH = Path("evals/baselines/baseline.json")
CANDIDATE_PATH = Path("evals/baselines/candidate.json")
RESULTS_DIR = Path("evals/results")

def get_git_sha() -> str:
    """Return current git commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"

def run_suite(
    api_base_url: str = "http://127.0.0.1:8000/api/v1",
    category_filter: str = None,
    output_path: Path = None,
    is_baseline: bool = False
) -> Dict[str, Any]:
    """Execute the full evaluation suite and generate snapshot report."""
    print("==================================================================")
    print("           Omni-Agent Evaluation Suite (CampusX / DeepEval)        ")
    print("==================================================================")

    # 1. Load Golden Test Cases
    cases = load_goldens(category=category_filter)
    print(f"Loaded {len(cases)} golden test cases from evals/goldens/goldens.jsonl")

    # 2. Setup isolated test threads
    rag_thread_id = f"eval-rag-{int(time.time())}"
    general_thread_id = f"eval-gen-{int(time.time())}"

    # Upload PDF fixture for RAG cases
    has_pdf = ensure_fixture_pdf_uploaded(api_base_url, rag_thread_id)
    if has_pdf:
        print(f"Uploaded test PDF fixture to thread: {rag_thread_id}")

    # 3. Execute all turns through the API
    executed_turns: List[Dict[str, Any]] = []
    latencies: List[float] = []
    errors_count = 0

    print("\nExecuting test queries through Omni-Agent API...")
    for idx, tc in enumerate(cases, 1):
        target_thread = rag_thread_id if tc.get("needs_pdf") else general_thread_id
        start_t = time.perf_counter()
        try:
            r = httpx.post(
                f"{api_base_url}/chat",
                json={"message": tc["input"], "thread_id": target_thread},
                timeout=30.0
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            latencies.append(elapsed_ms)

            if r.status_code == 200:
                data = r.json()
                executed_turns.append({
                    "id": tc["id"],
                    "category": tc["category"],
                    "input": tc["input"],
                    "expected_tools": tc.get("expected_tools", []),
                    "tools_used": data.get("tools_used", []),
                    "response": data.get("response", ""),
                    "expected_contains": tc.get("expected_contains", []),
                    "latency_ms": elapsed_ms,
                    "thread_id": target_thread
                })
                print(f"  [{idx:02d}/{len(cases):02d}] PASS: {tc['id']} ({elapsed_ms:.0f}ms) Tools: {data.get('tools_used')}")
            else:
                errors_count += 1
                executed_turns.append({
                    "id": tc["id"],
                    "category": tc["category"],
                    "input": tc["input"],
                    "expected_tools": tc.get("expected_tools", []),
                    "tools_used": [],
                    "response": "",
                    "expected_contains": tc.get("expected_contains", []),
                    "latency_ms": elapsed_ms,
                    "error": r.text
                })
                print(f"  [{idx:02d}/{len(cases):02d}] FAIL: {tc['id']} HTTP {r.status_code}")
        except Exception as exc:
            errors_count += 1
            print(f"  [{idx:02d}/{len(cases):02d}] ERROR: {tc['id']} - {exc}")

    # 4. Compute Metrics Across Tiers
    print("\nCalculating metrics across all 3 tiers...")

    # Tier 1: Component Evals
    rag_cases = [t for t in executed_turns if "rag" in t["category"]]
    retriever_metrics = eval_retriever_for_thread(rag_thread_id, rag_cases)
    generator_metrics = eval_generator_responses(executed_turns)

    # Tier 2: Pipeline RAG Triad
    pipeline_metrics = eval_rag_triad(rag_cases)

    # Tier 3: Application Quality, Safety, and Ops
    app_metrics = eval_application_quality(executed_turns)
    safety_turns = [t for t in executed_turns if "safety" in t["category"]]
    safety_metrics = eval_safety_gates(safety_turns)
    ops_metrics = eval_ops_metrics(latencies, errors_count, len(cases))

    # 5. Flatten into 3-level dotted IDs
    snapshot_metrics: Dict[str, float] = {}
    for k, v in retriever_metrics.items():
        snapshot_metrics[f"retriever.{k}"] = v
    for k, v in generator_metrics.items():
        snapshot_metrics[f"generator.{k}"] = v
    for k, v in pipeline_metrics.items():
        snapshot_metrics[f"pipeline.{k}"] = v
    for k, v in app_metrics.items():
        snapshot_metrics[f"application.{k}"] = v
    for k, v in safety_metrics.items():
        snapshot_metrics[f"safety.{k}"] = v
    for k, v in ops_metrics.items():
        snapshot_metrics[f"ops.{k}"] = v

    # 6. Format Executive Summary Table
    print("\n==================================================================")
    print("                       EVALUATION SCORECARD                       ")
    print("==================================================================")
    print(f"{'Metric ID':<42} {'Score':<8} {'Target':<8} {'Status':<6}")
    print("-" * 66)

    gate_failures = 0
    for m_id, val in snapshot_metrics.items():
        rule = rule_for(m_id)
        if rule.comparison == "gte":
            passed = val >= rule.threshold
        else:
            passed = val <= rule.threshold

        status_str = "✅ PASS" if passed else ("❌ FAIL" if rule.is_gate else "⚠️ WARN")
        if not passed and rule.is_gate:
            gate_failures += 1

        print(f"{m_id:<42} {val:<8.3f} {rule.threshold:<8.2f} {status_str}")

    print("-" * 66)
    if gate_failures == 0:
        print("RESULT: ALL SAFETY GATES & CORE TARGETS PASSED! 🚀")
    else:
        print(f"RESULT: ❌ {gate_failures} HARD RELEASE GATE(S) FAILED.")

    # 7. Write Snapshot JSON
    target_out = output_path or (BASELINE_PATH if is_baseline else CANDIDATE_PATH)
    target_out.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "metadata": {
            "timestamp": timestamp,
            "git_sha": get_git_sha(),
            "total_turns": len(cases),
            "errors_count": errors_count,
            "category_filter": category_filter,
            "target_out": str(target_out)
        },
        "metrics": snapshot_metrics,
        "records": executed_turns
    }

    with open(target_out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nSnapshot written to: {target_out}")

    # Also archive timestamped result
    ts_filename = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(RESULTS_DIR / ts_filename, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Archived historical report to: {RESULTS_DIR / ts_filename}")

    return snapshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Omni-Agent Evaluation Suite")
    parser.add_argument("--baseline", action="store_true", help="Save snapshot as baseline.json")
    parser.add_argument("--out", type=str, default=None, help="Custom output path")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    args = parser.parse_args()

    out_p = Path(args.out) if args.out else None
    run_suite(is_baseline=args.baseline, output_path=out_p, category_filter=args.category)
