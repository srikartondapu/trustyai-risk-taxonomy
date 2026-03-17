"""
TrustyAI Risk Taxonomy - Optimizer CSV Export

Generates the 4 CSV files consumed by the guardrail optimizer
(github.com/dahlem/guardrail-optimizer).

Usage:
    # Generate all 4 CSVs from a Garak scan report
    python export.py --report garak_results/phi3_scan.report.jsonl \
        --output optimizer_data/ --llm-id phi3-mini

    # Generate only risk dimensions (no scan data needed)
    python export.py --dimensions-only --output optimizer_data/

    # Include eval baseline scores in output summary
    python export.py --report garak_results/phi3_scan.report.jsonl \
        --eval-results eval_results/ --output optimizer_data/ --llm-id phi3-mini
"""

import argparse
import csv
import json
import os
from pathlib import Path

from taxonomy.query import TaxonomyManager
from parse_garak import parse_garak_report, to_demo_format


# The 12 optimizer risk dimensions.
# These are the optimizer's vocabulary -- we define all 12 here even though
# our taxonomy only has operational data (probes, detectors, evals) for 5.
OPTIMIZER_DIMENSIONS = [
    {"id": "harmful_content", "name": "Harmful Content Generation", "category": "safety",
     "description": "Model generates toxic, violent, or otherwise harmful content"},
    {"id": "jailbreak", "name": "Jailbreak Attacks", "category": "security",
     "description": "Model safety constraints circumvented through adversarial prompts"},
    {"id": "pii_leakage", "name": "PII Leakage", "category": "privacy",
     "description": "Model exposes personally identifiable information from training data or context"},
    {"id": "bias_fairness", "name": "Bias and Fairness", "category": "ethics",
     "description": "Model exhibits discriminatory behavior across demographic groups"},
    {"id": "hallucination", "name": "Hallucination", "category": "reliability",
     "description": "Model generates factually incorrect or fabricated information"},
    {"id": "prompt_injection", "name": "Prompt Injection", "category": "security",
     "description": "Model manipulated via malicious prompts to bypass safety measures"},
    {"id": "data_poisoning", "name": "Data Poisoning Vulnerability", "category": "security",
     "description": "Model susceptible to manipulation via poisoned training data"},
    {"id": "copyright_violation", "name": "Copyright Violation", "category": "legal",
     "description": "Model reproduces copyrighted material without authorization"},
    {"id": "misinformation", "name": "Misinformation Spread", "category": "safety",
     "description": "Model amplifies or generates misleading information"},
    {"id": "overreliance", "name": "Overreliance Risk", "category": "reliability",
     "description": "Users place excessive trust in model outputs without verification"},
    {"id": "context_overflow", "name": "Context Overflow", "category": "reliability",
     "description": "Model fails to properly handle long contexts or conversation history"},
    {"id": "api_abuse", "name": "API Abuse", "category": "security",
     "description": "Model endpoints exploited for denial of service or resource exhaustion"},
]

# Estimated guardrail costs. These are NOT measured -- they are rough estimates
# based on model size and type. Real values should be collected from production.
ESTIMATED_COSTS = {
    "toxicity-detector-hf": {
        "tokens": 100, "calls": 1, "memory_mb": 512,
        "latency_ms": 50, "cost_usd": 0.001, "carbon_kg": 0.0,
    },
    "jailbreak-detector-hf": {
        "tokens": 100, "calls": 1, "memory_mb": 512,
        "latency_ms": 50, "cost_usd": 0.001, "carbon_kg": 0.0,
    },
    "pii-detector-hf": {
        "tokens": 0, "calls": 1, "memory_mb": 256,
        "latency_ms": 20, "cost_usd": 0.0005, "carbon_kg": 0.0,
    },
    "granite-guardian-hap": {
        "tokens": 200, "calls": 1, "memory_mb": 1024,
        "latency_ms": 100, "cost_usd": 0.002, "carbon_kg": 0.0,
    },
}


def write_csv_file(filepath, rows, fieldnames):
    """Write rows to a CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_risk_dimensions(output_dir):
    """Generate risk_dimensions.csv -- static list of 12 optimizer dimensions."""
    filepath = os.path.join(output_dir, "risk_dimensions.csv")
    fieldnames = ["id", "name", "category", "description"]
    write_csv_file(filepath, OPTIMIZER_DIMENSIONS, fieldnames)
    return filepath, len(OPTIMIZER_DIMENSIONS)


def generate_risk_exposure(tm, report_path, output_dir, llm_id,
                           domain_id=None, use_case_id=None):
    """Generate risk_exposure_{llm_id}.csv from Garak scan data.

    Exposure = 1.0 - worst_pass_rate per dimension (conservative).
    Dimensions without scan data get exposure=1.0, confidence=0.0.

    When domain_id/use_case_id provided, uses profile thresholds to weight
    exposure by priority (critical dimensions get exposure amplified by 1.2x).
    """
    parsed = parse_garak_report(report_path)
    demo_results = to_demo_format(parsed)

    # Map probes to dimensions, track worst score per dimension
    dim_scores = {}
    dim_probe_counts = {}
    for result in demo_results:
        risk_id = tm.lookup_risk_by_probe(result["probe"])
        if not risk_id:
            continue
        dim_id = tm.get_dimension_for_risk(risk_id)
        if not dim_id:
            continue

        if dim_id not in dim_scores:
            dim_scores[dim_id] = []
        dim_scores[dim_id].append(result["score"])
        dim_probe_counts[dim_id] = dim_probe_counts.get(dim_id, 0) + 1

    # Build timestamp from scan data
    start_time = parsed["run_info"].get("start_time", "unknown")
    data_source_date = start_time[:10] if start_time != "unknown" else "unknown"

    # Resolve profile priorities if domain provided
    priority_weights = {}
    if domain_id:
        profile = tm.resolve_profile(domain_id, use_case_id)
        priority_map = {"critical": 1.2, "high": 1.1, "medium": 1.0, "low": 0.9}
        for did, priority in profile["dimension_priorities"].items():
            priority_weights[did] = priority_map.get(priority, 1.0)

    rows = []
    assessed_count = 0
    for dim in OPTIMIZER_DIMENSIONS:
        dim_id = dim["id"]
        if dim_id in dim_scores:
            worst_score = min(dim_scores[dim_id])
            exposure = round(1.0 - worst_score, 4)
            # Apply priority weighting if profile active
            if dim_id in priority_weights:
                exposure = round(min(1.0, exposure * priority_weights[dim_id]), 4)
            n_probes = dim_probe_counts[dim_id]
            confidence = round(min(0.90, 0.50 + n_probes * 0.10), 2)
            data_source = f"garak_scan_{data_source_date}"
            assessed_count += 1
        else:
            exposure = 1.0
            confidence = 0.0
            data_source = "not_assessed"

        rows.append({
            "risk_id": dim_id,
            "exposure": exposure,
            "confidence": confidence,
            "data_source": data_source,
        })

    filename = f"risk_exposure_{llm_id}.csv"
    filepath = os.path.join(output_dir, filename)
    fieldnames = ["risk_id", "exposure", "confidence", "data_source"]
    write_csv_file(filepath, rows, fieldnames)
    return filepath, assessed_count


def generate_guardrail_mitigation(tm, output_dir):
    """Generate guardrail_mitigation.csv from taxonomy detector mappings.

    Each active detector gets a row for its covered dimension.
    mitigation_score comes from effectiveness_score if measured, else blank.
    """
    rows = []
    for dim in tm.get_all_dimensions():
        risk_id = dim["primary_risk_id"]
        if not risk_id:
            continue
        dim_id = dim["id"]
        mitigation = tm.get_mitigations(risk_id)
        if not mitigation:
            continue

        for detector in mitigation.get("detectors", []):
            guardrail_id = detector["id"].replace("-", "_")
            eff = detector.get("effectiveness_score")
            # Optimizer requires mitigation_score when coverage=covered.
            # Only mark as "covered" when we have a real effectiveness score.
            if detector["deployment_status"] == "active" and eff is not None:
                coverage = "covered"
                mitigation_score = round(eff, 4)
            else:
                coverage = "not_checked"
                mitigation_score = ""

            rows.append({
                "guardrail_id": guardrail_id,
                "guardrail_name": detector["model_name"],
                "provider": "trustyai",
                "risk_id": dim_id,
                "coverage": coverage,
                "mitigation_score": mitigation_score,
            })

    filepath = os.path.join(output_dir, "guardrail_mitigation.csv")
    fieldnames = ["guardrail_id", "guardrail_name", "provider", "risk_id",
                  "coverage", "mitigation_score"]
    write_csv_file(filepath, rows, fieldnames)
    return filepath, len(rows)


def generate_guardrail_costs(tm, output_dir):
    """Generate guardrail_costs.csv with estimated costs.

    Values are ESTIMATED, not measured. Real latency and cost data
    should be collected from production monitoring.
    """
    rows = []
    seen = set()
    for dim in tm.get_all_dimensions():
        risk_id = dim["primary_risk_id"]
        if not risk_id:
            continue
        mitigation = tm.get_mitigations(risk_id)
        if not mitigation:
            continue

        for detector in mitigation.get("detectors", []):
            det_id = detector["id"]
            if det_id in seen:
                continue
            seen.add(det_id)

            guardrail_id = det_id.replace("-", "_")
            costs = ESTIMATED_COSTS.get(det_id, {
                "tokens": 0, "calls": 1, "memory_mb": 512,
                "latency_ms": 50, "cost_usd": 0.001, "carbon_kg": 0.0,
            })

            # Use real latency if available in taxonomy
            real_latency = detector.get("latency_ms")
            if real_latency is not None:
                costs["latency_ms"] = real_latency

            rows.append({
                "guardrail_id": guardrail_id,
                **costs,
            })

    filepath = os.path.join(output_dir, "guardrail_costs.csv")
    fieldnames = ["guardrail_id", "tokens", "calls", "memory_mb",
                  "latency_ms", "cost_usd", "carbon_kg"]
    write_csv_file(filepath, rows, fieldnames)
    return filepath, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimizer CSVs from taxonomy data"
    )
    parser.add_argument(
        "--report",
        help="Path to Garak .report.jsonl for exposure scores"
    )
    parser.add_argument(
        "--output", default="optimizer_data",
        help="Output directory (default: optimizer_data/)"
    )
    parser.add_argument(
        "--llm-id", default="phi3-mini",
        help="LLM identifier for exposure filename (default: phi3-mini)"
    )
    parser.add_argument(
        "--dimensions-only", action="store_true",
        help="Only generate risk_dimensions.csv (no scan data needed)"
    )
    parser.add_argument(
        "--eval-results",
        help="Path to eval_results/ directory to include baseline scores in summary"
    )
    parser.add_argument(
        "--domain",
        help="Domain profile ID for profile-weighted exposure (e.g., healthcare)"
    )
    parser.add_argument(
        "--use-case",
        help="Use-case ID within domain (e.g., billing_coding)"
    )
    args = parser.parse_args()

    if not args.dimensions_only and not args.report:
        parser.error("--report is required unless --dimensions-only is specified")

    os.makedirs(args.output, exist_ok=True)
    tm = TaxonomyManager()

    print(f"\nOptimizer CSV Export")
    print(f"{'='*60}")
    print(f"  Output: {args.output}/")

    # 1. Risk dimensions (always generated)
    path, count = generate_risk_dimensions(args.output)
    print(f"\n  risk_dimensions.csv")
    print(f"    {count} dimensions (static definitions)")

    if args.dimensions_only:
        print(f"\n  Done (dimensions only).")
        return

    # 2. Risk exposure
    path, assessed = generate_risk_exposure(
        tm, args.report, args.output, args.llm_id,
        domain_id=args.domain, use_case_id=getattr(args, "use_case", None),
    )
    not_assessed = len(OPTIMIZER_DIMENSIONS) - assessed
    print(f"\n  risk_exposure_{args.llm_id}.csv")
    print(f"    {assessed} from Garak scan (real data)")
    print(f"    {not_assessed} not assessed (exposure=1.0)")

    # 3. Guardrail mitigation
    path, count = generate_guardrail_mitigation(tm, args.output)
    # Check how many have effectiveness scores
    eff_count = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mitigation_score"):
                eff_count += 1
    print(f"\n  guardrail_mitigation.csv")
    print(f"    {count} guardrail-risk mappings")
    if eff_count > 0:
        print(f"    {eff_count} with measured effectiveness")
    else:
        print(f"    effectiveness NOT measured (run compare_scans.py first)")

    # 4. Guardrail costs
    path, count = generate_guardrail_costs(tm, args.output)
    print(f"\n  guardrail_costs.csv")
    print(f"    {count} guardrails (ESTIMATED costs, not measured)")

    # 5. Eval baselines (informational, not a CSV the optimizer needs)
    if args.eval_results:
        eval_dir = Path(args.eval_results)
        if eval_dir.exists():
            print(f"\n  Eval baselines (from {args.eval_results}):")
            for result_file in sorted(eval_dir.rglob("results_*.json")):
                with open(result_file) as f:
                    data = json.load(f)
                for task_name, results in data.get("results", {}).items():
                    for k, v in results.items():
                        if not k.endswith("_stderr") and k != "alias":
                            metric = k.replace(",none", "")
                            dim = result_file.parent.parent.name
                            if isinstance(v, (int, float)):
                                print(f"    [{dim:>20s}] {task_name}: {metric} = {v:.4f}")
                            else:
                                print(f"    [{dim:>20s}] {task_name}: {metric} = {v}")

    print(f"\n{'='*60}")
    print(f"  DATA PROVENANCE:")
    print(f"    Garak scan:       {args.report}")
    print(f"    Taxonomy:         taxonomy/data/ (5 operational dimensions)")
    print(f"    Cost data:        ESTIMATED (not measured)")
    if args.eval_results:
        print(f"    Eval baselines:   {args.eval_results}")
    print()


if __name__ == "__main__":
    main()
