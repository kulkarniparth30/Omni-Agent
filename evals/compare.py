"""
evals/compare.py

Compares two evaluation snapshots (e.g. baseline.json vs candidate.json)
and produces a regression analysis report.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evals.metric_registry import rule_for

DEFAULT_BASELINE = Path("evals/baselines/baseline.json")
DEFAULT_CANDIDATE = Path("evals/baselines/candidate.json")

def load_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_snapshots(baseline_path: Path, candidate_path: Path):
    """Compare baseline vs candidate and print regression diff."""
    base = load_snapshot(baseline_path)
    cand = load_snapshot(candidate_path)

    base_metrics = base.get("metrics", {})
    cand_metrics = cand.get("metrics", {})

    print("==================================================================")
    print("           Omni-Agent Evaluation Regression Report                ")
    print(f"  Baseline:  {baseline_path} (Git: {base.get('metadata', {}).get('git_sha', '?')})")
    print(f"  Candidate: {candidate_path} (Git: {cand.get('metadata', {}).get('git_sha', '?')})")
    print("==================================================================")
    print(f"{'Metric ID':<40} {'Baseline':<10} {'Candidate':<10} {'Delta':<8} {'Status'}")
    print("-" * 75)

    all_keys = sorted(list(set(base_metrics.keys()).union(cand_metrics.keys())))
    regressions = 0

    for m_id in all_keys:
        b_val = base_metrics.get(m_id)
        c_val = cand_metrics.get(m_id)
        rule = rule_for(m_id)

        if b_val is None:
            print(f"{m_id:<40} {'NEW':<10} {c_val:<10.3f} {'-':<8} 🆕")
            continue
        if c_val is None:
            print(f"{m_id:<40} {b_val:<10.3f} {'MISSING':<10} {'-':<8} ❌ REMOVED")
            regressions += 1
            continue

        delta = c_val - b_val
        # Determine if delta is good or bad based on rule
        if rule.comparison == "gte":
            # Higher is better
            if delta > 0.005:
                status = "🔼 IMPROVED"
            elif delta < -0.01:
                status = "🔻 REGRESSED"
                regressions += 1
            else:
                status = "⏹️ SAME"
        else:
            # Lower is better (e.g. latency, error rate, toxicity)
            if delta < -5.0:
                status = "🔼 IMPROVED"
            elif delta > 10.0:
                status = "🔻 REGRESSED"
                regressions += 1
            else:
                status = "⏹️ SAME"

        delta_str = f"{delta:+.3f}" if abs(delta) < 100 else f"{delta:+.0f}"
        print(f"{m_id:<40} {b_val:<10.3f} {c_val:<10.3f} {delta_str:<8} {status}")

    print("-" * 75)
    if regressions == 0:
        print("RESULT: 🎉 NO REGRESSIONS DETECTED! Candidate is safe to deploy.")
    else:
        print(f"RESULT: ⚠️ {regressions} METRIC REGRESSION(S) DETECTED! Review before deploying.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two eval snapshots")
    parser.add_argument("--base", type=str, default=str(DEFAULT_BASELINE), help="Path to baseline.json")
    parser.add_argument("--cand", type=str, default=str(DEFAULT_CANDIDATE), help="Path to candidate.json")
    args = parser.parse_args()

    compare_snapshots(Path(args.base), Path(args.cand))
