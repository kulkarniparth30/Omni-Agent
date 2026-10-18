"""
evals/harness.py

Evaluation test harness and golden dataset loader.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

GOLDENS_PATH = Path("evals/goldens/goldens.jsonl")
FIXTURE_PDF_PATH = Path("evals/goldens/test_fixture.pdf")

def load_goldens(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load test cases from goldens.jsonl, optionally filtered by category."""
    if not GOLDENS_PATH.exists():
        raise FileNotFoundError(f"Goldens dataset not found at {GOLDENS_PATH}")

    cases = []
    with open(GOLDENS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if category and item.get("category") != category:
                continue
            cases.append(item)
    return cases

def ensure_fixture_pdf_uploaded(api_base_url: str, thread_id: str) -> bool:
    """Upload the golden test fixture PDF to the target thread via the API."""
    if not FIXTURE_PDF_PATH.exists():
        return False

    with open(FIXTURE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    files = {"file": ("test_fixture.pdf", pdf_bytes, "application/pdf")}
    try:
        r = httpx.post(f"{api_base_url}/threads/{thread_id}/documents", files=files, timeout=30.0)
        return r.status_code == 200
    except Exception:
        return False
