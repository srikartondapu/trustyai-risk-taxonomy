"""
TrustyAI Risk Taxonomy - Garak Report Parser

Parses real Garak .report.jsonl files into the format expected by
the governance loop demo script.

Usage:
    from parse_garak import parse_garak_report

    results = parse_garak_report("path/to/report.jsonl")
    # Returns list of dicts matching demo script format

Or standalone:
    python parse_garak.py path/to/report.jsonl
"""

import json
import sys
from pathlib import Path


def parse_garak_report(report_path: str) -> dict:
    """
    Parse a Garak .report.jsonl file.

    Returns:
        {
            "run_info": {
                "garak_version": str,
                "start_time": str,
                "run_id": str,
                "model_name": str,
                "probes_requested": str,
                "generations": int,
            },
            "eval_results": [
                {
                    "probe": str,          # e.g. "dan.DanInTheWild"
                    "detector": str,       # e.g. "unsafe_content.ToxicCommentModel"
                    "passed": int,
                    "total": int,
                    "score": float,        # passed / total
                },
                ...
            ],
            "probe_summary": {
                "probe_name": {
                    "detectors": int,
                    "avg_score": float,
                    "worst_score": float,
                    "best_score": float,
                    "total_passed": int,
                    "total_total": int,
                },
                ...
            }
        }
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    run_info = {}
    eval_results = []

    with open(report_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("entry_type", "")

            # Parse run setup
            if entry_type == "start_run setup":
                run_info = {
                    "garak_version": entry.get("_config.version", "unknown"),
                    "start_time": entry.get("transient.starttime_iso", "unknown"),
                    "run_id": entry.get("transient.run_id", "unknown"),
                    "model_name": entry.get("plugins.target_name", "unknown"),
                    "probes_requested": entry.get("plugins.probe_spec", "unknown"),
                    "generations": entry.get("run.generations", 0),
                }

            # Parse eval entries -- these have the aggregated scores
            elif entry_type == "eval":
                passed = entry.get("passed", 0)
                total = entry.get("total", 0)
                score = passed / total if total > 0 else 0.0

                eval_results.append({
                    "probe": entry.get("probe", "unknown"),
                    "detector": entry.get("detector", "unknown"),
                    "passed": passed,
                    "total": total,
                    "score": score,
                })

    # Build probe summary
    probe_summary = {}
    for r in eval_results:
        probe = r["probe"]
        if probe not in probe_summary:
            probe_summary[probe] = {
                "detectors": 0,
                "scores": [],
                "total_passed": 0,
                "total_total": 0,
            }
        probe_summary[probe]["detectors"] += 1
        probe_summary[probe]["scores"].append(r["score"])
        probe_summary[probe]["total_passed"] += r["passed"]
        probe_summary[probe]["total_total"] += r["total"]

    for probe, data in probe_summary.items():
        scores = data.pop("scores")
        data["avg_score"] = sum(scores) / len(scores) if scores else 0.0
        data["worst_score"] = min(scores) if scores else 0.0
        data["best_score"] = max(scores) if scores else 0.0

    return {
        "run_info": run_info,
        "eval_results": eval_results,
        "probe_summary": probe_summary,
    }


def to_demo_format(parsed: dict) -> list[dict]:
    """
    Convert parsed Garak results to the format expected by
    demo_governance_loop.py.

    Groups by probe and uses the worst detector score per probe
    (conservative -- if any detector flagged it, it's a concern).
    """
    probe_scores = {}

    for r in parsed["eval_results"]:
        probe = r["probe"]
        if probe not in probe_scores:
            probe_scores[probe] = {
                "probe": probe,
                "passed": 0,
                "total": 0,
                "detector_scores": [],
            }
        probe_scores[probe]["detector_scores"].append(r["score"])
        probe_scores[probe]["passed"] += r["passed"]
        probe_scores[probe]["total"] += r["total"]

    results = []
    for probe, data in probe_scores.items():
        # Use worst (minimum) detector score -- conservative approach
        worst_score = min(data["detector_scores"])
        results.append({
            "probe": probe,
            "detector": "garak.auto",  # Garak uses auto-detection
            "passed": data["passed"],
            "total": data["total"],
            "score": worst_score,
        })

    return results


def print_summary(parsed: dict):
    """Print a human-readable summary of the scan."""
    info = parsed["run_info"]
    print(f"Garak Scan Report")
    print(f"  Version:    {info.get('garak_version', '?')}")
    print(f"  Run ID:     {info.get('run_id', '?')}")
    print(f"  Started:    {info.get('start_time', '?')}")
    print(f"  Model:      {info.get('model_name', '?')}")
    print(f"  Probes:     {info.get('probes_requested', '?')}")
    print(f"  Generations: {info.get('generations', '?')}")

    print(f"\nEval Results ({len(parsed['eval_results'])} entries):")
    print(f"  {'Probe':>45} | {'Detector':>45} | {'Score':>10}")
    print(f"  {'-'*45} | {'-'*45} | {'-'*10}")
    for r in parsed["eval_results"]:
        icon = "FAIL" if r["score"] < 0.5 else "WARN" if r["score"] < 0.8 else "PASS"
        print(f"  {r['probe']:>45} | {r['detector']:>45} | {icon} {r['passed']}/{r['total']} ({r['score']:.0%})")

    print(f"\nProbe Summary:")
    for probe, data in parsed["probe_summary"].items():
        icon = "FAIL" if data["worst_score"] < 0.5 else "WARN" if data["worst_score"] < 0.8 else "PASS"
        print(f"  {icon} {probe}: avg {data['avg_score']:.0%}, worst {data['worst_score']:.0%} ({data['detectors']} detectors)")

    # Demo format preview
    demo = to_demo_format(parsed)
    print(f"\nDemo Format ({len(demo)} probes):")
    for d in demo:
        icon = "FAIL" if d["score"] < 0.5 else "WARN" if d["score"] < 0.8 else "PASS"
        print(f"  {icon} {d['probe']:>45} | {d['score']:.0%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_garak.py <report.jsonl>")
        print("\nSearching for report files...")
        # Try to find report files
        default_dir = Path.home() / ".local/share/garak/garak_runs"
        if default_dir.exists():
            reports = sorted(default_dir.glob("*.report.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            for r in reports[:5]:
                size = r.stat().st_size / 1024
                print(f"  {r.name} ({size:.0f}KB)")
        sys.exit(1)

    report_path = sys.argv[1]
    parsed = parse_garak_report(report_path)
    print_summary(parsed)
