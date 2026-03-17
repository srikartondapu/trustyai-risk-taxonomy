"""
Tests for export.py, compare_scans.py, and taxonomy write-back.

Run: python test_export.py
"""

import csv
import json
import os
import sys
import tempfile
import shutil

from taxonomy.query import TaxonomyManager
from export import (
    generate_risk_dimensions,
    generate_risk_exposure,
    generate_guardrail_mitigation,
    generate_guardrail_costs,
)
from compare_scans import (
    map_probes_to_dimensions,
    compute_effectiveness,
)

passed = 0
failed = 0


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check(condition, msg):
    global passed, failed
    if condition:
        print(f"  PASS  {msg}")
        passed += 1
    else:
        print(f"  FAIL  {msg}")
        failed += 1


def main():
    tm = TaxonomyManager()
    output_dir = tempfile.mkdtemp(prefix="test_export_")

    try:
        # ==============================================================
        # TEST 1: Risk dimensions CSV
        # ==============================================================
        section("TEST 1: Risk Dimensions CSV")
        path, count = generate_risk_dimensions(output_dir)
        check(count == 12, f"12 dimensions (got {count})")
        check(os.path.exists(path), "File created")

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        check(len(rows) == 12, f"12 rows in CSV (got {len(rows)})")
        cols = set(rows[0].keys())
        check(cols == {"id", "name", "category", "description"}, f"Correct columns: {cols}")

        dim_ids = {r["id"] for r in rows}
        for expected in ["harmful_content", "jailbreak", "pii_leakage", "bias_fairness", "hallucination"]:
            check(expected in dim_ids, f"  Contains {expected}")

        # ==============================================================
        # TEST 2: Risk exposure CSV
        # ==============================================================
        section("TEST 2: Risk Exposure CSV")
        report_path = "garak_results/phi3_scan.report.jsonl"
        if os.path.exists(report_path):
            path, assessed = generate_risk_exposure(tm, report_path, output_dir, "test-model")
            check(assessed == 5, f"5 assessed dimensions (got {assessed})")
            check("risk_exposure_test-model.csv" in path, "Filename includes llm-id")

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            check(len(rows) == 12, f"12 rows (all dimensions, got {len(rows)})")

            # Check assessed dimensions have real data
            for row in rows:
                if row["data_source"] != "not_assessed":
                    exposure = float(row["exposure"])
                    confidence = float(row["confidence"])
                    check(0 <= exposure <= 1.0, f"  {row['risk_id']}: exposure={exposure} in [0,1]")
                    check(confidence > 0, f"  {row['risk_id']}: confidence={confidence} > 0")

            # Check unassessed dimensions
            unassessed = [r for r in rows if r["data_source"] == "not_assessed"]
            check(len(unassessed) == 7, f"7 unassessed dimensions (got {len(unassessed)})")
            for row in unassessed:
                check(float(row["exposure"]) == 1.0, f"  {row['risk_id']}: exposure=1.0 (conservative)")
                check(float(row["confidence"]) == 0.0, f"  {row['risk_id']}: confidence=0.0")
        else:
            print(f"  SKIP  No scan report at {report_path}")

        # ==============================================================
        # TEST 3: Guardrail mitigation CSV
        # ==============================================================
        section("TEST 3: Guardrail Mitigation CSV")
        path, count = generate_guardrail_mitigation(tm, output_dir)
        check(count >= 3, f"At least 3 guardrail mappings (got {count})")

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        cols = set(rows[0].keys())
        expected_cols = {"guardrail_id", "guardrail_name", "provider", "risk_id",
                         "coverage", "mitigation_score"}
        check(cols == expected_cols, f"Correct columns")

        # Check that measured effectiveness is present (3 measured)
        with_scores = [r for r in rows if r["mitigation_score"]]
        check(len(with_scores) >= 3, f"{len(with_scores)} detectors with measured effectiveness")

        # Verify specific detector scores
        for row in rows:
            if row["guardrail_id"] == "toxicity_detector_hf":
                check(float(row["mitigation_score"]) == 1.0, "  Toxicity effectiveness = 1.0")
            if row["guardrail_id"] == "jailbreak_detector_hf":
                jailbreak_eff = float(row["mitigation_score"])
                check(0.85 <= jailbreak_eff <= 0.95, f"  Jailbreak effectiveness = {jailbreak_eff:.4f} (block-rate corrected)")
            if row["guardrail_id"] == "pii_detector_hf":
                check(float(row["mitigation_score"]) == 1.0, "  PII effectiveness = 1.0")

        # ==============================================================
        # TEST 4: Guardrail costs CSV
        # ==============================================================
        section("TEST 4: Guardrail Costs CSV")
        path, count = generate_guardrail_costs(tm, output_dir)
        check(count >= 3, f"At least 3 guardrails (got {count})")

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        expected_cols = {"guardrail_id", "tokens", "calls", "memory_mb",
                         "latency_ms", "cost_usd", "carbon_kg"}
        check(set(rows[0].keys()) == expected_cols, "Correct cost columns")

        # No duplicate guardrails
        ids = [r["guardrail_id"] for r in rows]
        check(len(ids) == len(set(ids)), "No duplicate guardrail entries")

        # ==============================================================
        # TEST 5: Effectiveness computation
        # ==============================================================
        section("TEST 5: Effectiveness Computation")

        # Case 1: Risk reduced from 60% exposure to 10%
        before = {"dim_a": [{"probe": "p1", "score": 0.4, "passed": 4, "total": 10, "risk_id": "r1"}]}
        after = {"dim_a": [{"probe": "p1", "score": 0.9, "passed": 9, "total": 10, "risk_id": "r1"}]}
        result = compute_effectiveness(before, after)
        eff = result["dim_a"]["effectiveness"]
        check(abs(eff - 0.8333) < 0.01, f"60%->10% exposure = ~83% effectiveness (got {eff:.4f})")

        # Case 2: Already safe, stays safe
        before = {"dim_b": [{"probe": "p1", "score": 1.0, "passed": 10, "total": 10, "risk_id": "r1"}]}
        after = {"dim_b": [{"probe": "p1", "score": 1.0, "passed": 10, "total": 10, "risk_id": "r1"}]}
        result = compute_effectiveness(before, after)
        check(result["dim_b"]["effectiveness"] == 1.0, "Already safe -> effectiveness = 1.0")

        # Case 3: Got worse (like our jailbreak results)
        before = {"dim_c": [{"probe": "p1", "score": 0.57, "passed": 57, "total": 100, "risk_id": "r1"}]}
        after = {"dim_c": [{"probe": "p1", "score": 0.0, "passed": 0, "total": 100, "risk_id": "r1"}]}
        result = compute_effectiveness(before, after)
        check(result["dim_c"]["effectiveness"] == 0.0, "Got worse -> effectiveness clamped to 0.0")

        # Case 4: Full fix (100% exposure -> 0% exposure)
        before = {"dim_d": [{"probe": "p1", "score": 0.0, "passed": 0, "total": 10, "risk_id": "r1"}]}
        after = {"dim_d": [{"probe": "p1", "score": 1.0, "passed": 10, "total": 10, "risk_id": "r1"}]}
        result = compute_effectiveness(before, after)
        check(result["dim_d"]["effectiveness"] == 1.0, "Full fix -> effectiveness = 1.0")

        # ==============================================================
        # TEST 6: Probe-to-dimension mapping
        # ==============================================================
        section("TEST 6: Probe-to-Dimension Mapping")
        probe_results = [
            {"probe": "dan.DanInTheWild", "score": 0.57, "passed": 57, "total": 100},
            {"probe": "lmrc.Bullying", "score": 1.0, "passed": 10, "total": 10},
            {"probe": "leakreplay.GuardianCloze", "score": 0.96, "passed": 96, "total": 100},
            {"probe": "unknown.Probe", "score": 0.5, "passed": 5, "total": 10},
        ]
        dim_results = map_probes_to_dimensions(probe_results, tm)
        check("jailbreak" in dim_results, "DAN mapped to jailbreak")
        check("harmful_content" in dim_results, "Bullying mapped to harmful_content")
        check("pii_leakage" in dim_results, "GuardianCloze mapped to pii_leakage")
        check("unknown" not in str(dim_results), "Unknown probe excluded")

        # ==============================================================
        # TEST 7: Write-back and reload
        # ==============================================================
        section("TEST 7: Write-back and Reload")

        # Copy taxonomy data to temp dir for isolated testing
        test_data_dir = os.path.join(output_dir, "taxonomy_data")
        shutil.copytree(tm._data_dir, test_data_dir)
        test_tm = TaxonomyManager(data_dir=test_data_dir)

        # Read initial value
        detectors = test_tm.get_active_detectors("atlas-spreading-toxicity")
        initial_eff = detectors[0].get("effectiveness_score")

        # Write new value
        test_tm.update_detector_effectiveness("toxicity-detector-hf", 0.95, latency_ms=42.5)

        # Verify after reload
        test_tm2 = TaxonomyManager(data_dir=test_data_dir)
        detectors = test_tm2.get_active_detectors("atlas-spreading-toxicity")
        check(detectors[0]["effectiveness_score"] == 0.95, "Effectiveness written and reloaded")
        check(detectors[0]["latency_ms"] == 42.5, "Latency written and reloaded")

        # Verify unknown detector raises
        try:
            test_tm.update_detector_effectiveness("nonexistent-detector", 0.5)
            check(False, "Should raise ValueError for unknown detector")
        except ValueError:
            check(True, "ValueError raised for unknown detector")

        # ==============================================================
        # SUMMARY
        # ==============================================================
        print(f"\n{'=' * 60}")
        print(f"  {passed} PASSED, {failed} FAILED")
        print(f"{'=' * 60}")

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
