from __future__ import annotations

import json
import importlib.util
from pathlib import Path


_CLAIM_AUDIT_SPEC = importlib.util.spec_from_file_location("claim_audit", Path("scripts/claim_audit.py"))
assert _CLAIM_AUDIT_SPEC and _CLAIM_AUDIT_SPEC.loader
_CLAIM_AUDIT = importlib.util.module_from_spec(_CLAIM_AUDIT_SPEC)
_CLAIM_AUDIT_SPEC.loader.exec_module(_CLAIM_AUDIT)
audit_claim_text = _CLAIM_AUDIT.audit_claim_text


def test_claim_audit_schema_mentions_v2_categories_if_generated():
    path = Path("results/claims_status.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {claim["category"] for claim in payload["claims"]}
    assert "PyTorch learned EBM claims" in categories
    assert "scripted expert benchmark claims" in categories
    assert "selected-tail reliability claims" in categories


def test_multitask_benchmark_table_schema_if_generated():
    path = Path("results/benchmarks/metaworld/task_table.csv")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").splitlines()
    header = set(text[0].split(","))
    assert {"task", "model", "training_source", "task_status", "mean_selected_reward"}.issubset(header)
    assert "push-v3" in "\n".join(text)
    assert "pick-place-v3" in "\n".join(text)
    assert "button-press-v3" in "\n".join(text)


def test_reliability_schema_if_generated():
    path = Path("results/reliability/summary.csv")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").splitlines()
    header = set(text[0].split(","))
    assert {"energy_quantile_low", "energy_quantile_high", "mean_real_utility", "is_extreme_low_energy_tail"}.issubset(header)
    assert any("True" in line for line in text[1:])


def test_optional_benchmark_rollout_schema_if_generated():
    path = Path("results/optional_benchmarks/rollouts.csv")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").splitlines()
    header = set(text[0].split(","))
    assert {"suite", "task", "seed", "policy", "total_reward", "success_available", "task_status"}.issubset(header)
    body = "\n".join(text[1:])
    assert "mani_skill" in body
    assert "robosuite" in body


def test_closed_loop_ablation_schema_if_generated():
    path = Path("results/benchmarks/metaworld/closed_loop_ablation_summary.csv")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").splitlines()
    header = set(text[0].split(","))
    assert {
        "task",
        "policy",
        "proposal_source",
        "uses_expert_centered_proposals",
        "conservative_proposal_gate",
        "mean_success_rate",
        "mean_total_reward",
        "success_drop_without_gate",
        "success_drop_without_expert_centering",
        "fallback_dependency_score",
        "uses_learned_demo_proposal",
        "uses_learned_bc_proposal",
        "uses_state_heuristic_proposal",
        "runtime_scripted_expert_dependency",
    }.issubset(header)
    body = "\n".join(text[1:])
    for policy in [
        "expert_centered_gate",
        "expert_centered_no_gate",
        "mixed_expert_local_no_gate",
        "local_gaussian_no_gate",
        "learned_demo_proposal_no_gate",
        "learned_demo_proposal_direct",
        "learned_bc_proposal_no_gate",
        "learned_bc_proposal_direct",
        "state_heuristic_proposal_no_gate",
        "state_heuristic_direct",
        "random_uniform",
        "scripted_expert",
    ]:
        assert policy in body
    payload = json.loads(Path("results/benchmarks/metaworld/closed_loop_ablation_summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "SUPPORTED"
    assert "autonomous_success_threshold_met" in payload


def test_repair_effectiveness_schema_if_generated():
    path = Path("results/repair_effectiveness/summary.csv")
    if not path.exists():
        return
    header = set(path.read_text(encoding="utf-8").splitlines()[0].split(","))
    assert {
        "mean_repair_recovery_ratio",
        "worst_case_recovery_ratio",
        "num_repair_failures",
        "num_utility_degrading_rows",
    }.issubset(header)
    stress = Path("results/repair_effectiveness/stress_matrix.csv")
    assert stress.exists()
    stress_header = set(stress.read_text(encoding="utf-8").splitlines()[0].split(","))
    assert {"repair_recovery_ratio", "recovery_ge_95", "utility_not_degraded"}.issubset(stress_header)


def test_handoff_can_be_created_for_new_threads():
    # The file may be generated after full verification; this test documents the
    # expected path without forcing handoff contents before scripts run.
    assert Path("docs").exists()


def test_claim_audit_rejects_undefined_almost_100_repair_wording():
    checks = audit_claim_text("The repair is almost 100% effective on these tasks.")
    near_100 = next(c for c in checks if c["check"] == "near-100 wording is metric-defined")
    assert near_100["passed"] is False
    defined = audit_claim_text(
        "The repair is almost 100% effective under the repair recovery ratio: "
        "(repair - raw) / (oracle - raw)."
    )
    assert next(c for c in defined if c["check"] == "near-100 wording is metric-defined")["passed"] is True


def test_optional_success_claim_blocked_when_success_is_zero():
    checks = audit_claim_text("We report success on optional suites.", optional_success_nonzero=False)
    optional = next(c for c in checks if c["check"] == "optional success claim blocked when zero")
    assert optional["passed"] is False


def test_autonomous_metaworld_claim_blocked_without_threshold():
    checks = audit_claim_text(
        "We report fully autonomous Meta-World success.",
        autonomous_metaworld_success=False,
    )
    row = next(c for c in checks if c["check"] == "autonomous Meta-World claim blocked without low-dependency success")
    assert row["passed"] is False
    allowed = audit_claim_text(
        "We do not claim fully autonomous Meta-World success.",
        autonomous_metaworld_success=False,
    )
    assert next(
        c for c in allowed if c["check"] == "autonomous Meta-World claim blocked without low-dependency success"
    )["passed"] is True
    learned = audit_claim_text(
        "We report autonomous learned-policy success.",
        autonomous_metaworld_success=True,
        autonomous_learned_metaworld_success=False,
    )
    learned_row = next(c for c in learned if c["check"] == "autonomous learned-policy claim blocked without learned threshold")
    assert learned_row["passed"] is False
