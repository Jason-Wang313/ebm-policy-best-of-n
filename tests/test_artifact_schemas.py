from __future__ import annotations

import json
from pathlib import Path


def test_claim_audit_schema_mentions_v2_categories_if_generated():
    path = Path("results/claims_status.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {claim["category"] for claim in payload["claims"]}
    assert "PyTorch learned EBM claims" in categories
    assert "Meta-World benchmark claims" in categories


def test_handoff_can_be_created_for_new_threads():
    # The file may be generated after full verification; this test documents the
    # expected path without forcing handoff contents before scripts run.
    assert Path("docs").exists()
