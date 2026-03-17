"""
Test script for TrustyAI Risk Taxonomy Query Layer.

Run: python test_taxonomy.py
"""

from taxonomy.query import TaxonomyManager


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    tm = TaxonomyManager()

    # ============================================================
    # TEST 1: Taxonomy stats
    # ============================================================
    section("TEST 1: Taxonomy Stats")
    stats = tm.get_taxonomy_stats()
    print(f"Total risks:            {stats['total_risks']}")
    print(f"Operational dimensions: {stats['operational_dimensions']}")
    print(f"Risks with probes:      {stats['risks_with_garak_probes']}")
    print(f"Risks with detectors:   {stats['risks_with_detectors']}")
    print(f"Risks with evals:       {stats['risks_with_evaluations']}")
    print(f"Total probes mapped:    {stats['total_probes_mapped']}")
    print(f"Taxonomies: {list(stats['taxonomy_counts'].keys())}")
    assert stats["total_risks"] == 546, f"Expected 546 risks, got {stats['total_risks']}"
    assert stats["operational_dimensions"] == 5
    print("[PASS] Stats correct")

    # ============================================================
    # TEST 2: Risk lookup from 546
    # ============================================================
    section("TEST 2: Risk Lookup (546 risks)")
    risk = tm.get_risk("atlas-jailbreaking")
    assert risk is not None
    assert risk["name"] == "Jailbreaking"
    assert risk["isDefinedByTaxonomy"] == "ibm-risk-atlas"
    print(f"  {risk['id']}: {risk['name']} [{risk['isDefinedByTaxonomy']}]")

    # Check a non-IBM risk
    nist = tm.get_risk("nist-confabulation")
    assert nist is not None
    assert nist["isDefinedByTaxonomy"] == "nist-ai-rmf"
    assert "atlas-hallucination" in nist.get("exact_mappings", [])
    print(f"  {nist['id']}: {nist['name']} (exact_mappings: {nist['exact_mappings']})")

    owasp = tm.get_risk("llm01-prompt-injection")
    assert owasp is not None
    print(f"  {owasp['id']}: {owasp['name']}")
    print("[PASS] Risks from multiple taxonomies accessible")

    # ============================================================
    # TEST 3: Filter by taxonomy
    # ============================================================
    section("TEST 3: Risks by Taxonomy")
    for tax in ["ibm-risk-atlas", "nist-ai-rmf", "owasp-llm-2.0"]:
        risks = tm.get_risks_by_taxonomy(tax)
        print(f"  {tax}: {len(risks)} risks")
    assert len(tm.get_risks_by_taxonomy("ibm-risk-atlas")) == 99
    assert len(tm.get_risks_by_taxonomy("owasp-llm-2.0")) == 10
    print("[PASS] Taxonomy filtering works")

    # ============================================================
    # TEST 4: Cross-taxonomy mappings
    # ============================================================
    section("TEST 4: Cross-Taxonomy Mappings")
    related = tm.get_related_risks("atlas-jailbreaking")
    print(f"  atlas-jailbreaking broad_mappings:")
    for r in related["broad_mappings"]:
        print(f"    -> {r['id']}: {r['name']}")
    assert len(related["broad_mappings"]) == 2  # nist + owasp
    print(f"  atlas-jailbreaking related_mappings: {len(related['related_mappings'])} risks")
    print("[PASS] Cross-taxonomy resolution works")

    # ============================================================
    # TEST 5: All dimensions
    # ============================================================
    section("TEST 5: Operational Dimensions")
    dims = tm.get_all_dimensions()
    assert len(dims) == 5
    for d in dims:
        status = "[PASS]" if d["has_detector"] else "[FAIL]"
        print(f"  {status} {d['id']:20} -> {d['primary_risk_id']}")
    print("[PASS] 5 dimensions loaded")

    # ============================================================
    # TEST 6: Dimension for risk (reverse lookup)
    # ============================================================
    section("TEST 6: Risk -> Dimension Reverse Lookup")
    tests = [
        ("atlas-spreading-toxicity", "harmful_content"),
        ("atlas-jailbreaking", "jailbreak"),
        ("atlas-exposing-personal-information", "pii_leakage"),
        ("atlas-decision-bias", "bias_fairness"),
        ("atlas-hallucination", "hallucination"),
        ("atlas-data-poisoning", None),  # Not an operational dimension
    ]
    for risk_id, expected_dim in tests:
        actual = tm.get_dimension_for_risk(risk_id)
        status = "[PASS]" if actual == expected_dim else "[FAIL]"
        print(f"  {status} {risk_id:45} -> {actual or 'NO DIMENSION'}")
        assert actual == expected_dim, f"Expected {expected_dim}, got {actual}"
    print("[PASS] Reverse lookup correct")

    # ============================================================
    # TEST 7: Garak probes with priority filter
    # ============================================================
    section("TEST 7: Garak Probes")
    all_probes = tm.get_garak_probes("atlas-jailbreaking")
    high_probes = tm.get_garak_probes("atlas-jailbreaking", min_priority="HIGH")
    print(f"  Jailbreak all probes:  {len(all_probes)}")
    print(f"  Jailbreak HIGH only:   {len(high_probes)}")
    assert len(all_probes) == 6
    assert len(high_probes) == 3
    for p in high_probes:
        print(f"    {p['probe']}")
    print("[PASS] Priority filtering works")

    # ============================================================
    # TEST 8: Probe reverse lookup
    # ============================================================
    section("TEST 8: Probe -> Risk Reverse Lookup")
    tests = [
        ("dan.DanInTheWild", "atlas-jailbreaking"),
        ("realtoxicityprompts.RTPBlank", "atlas-spreading-toxicity"),
        ("leakreplay.GuardianCloze", "atlas-exposing-personal-information"),
        ("packagehallucination.Python", "atlas-hallucination"),
        ("some.UnknownProbe", None),
    ]
    for probe, expected in tests:
        actual = tm.lookup_risk_by_probe(probe)
        print(f"  {probe:>40} -> {actual or 'NOT MAPPED'}")
        assert actual == expected
    print("[PASS] Probe reverse lookup correct")

    # ============================================================
    # TEST 9: Batch probe lookup (simulating Garak scan)
    # ============================================================
    section("TEST 9: Batch Probe Lookup (Garak Scan Simulation)")
    scan_results = [
        "dan.DanInTheWild", "dan.AutoDANCached",
        "realtoxicityprompts.RTPBlank", "lmrc.Bullying",
        "packagehallucination.Python",
    ]
    grouped = tm.lookup_risks_by_probes(scan_results)
    print(f"  Scan found {len(scan_results)} probe hits across {len(grouped)} risks:")
    for risk_id, probes in grouped.items():
        risk = tm.get_risk(risk_id)
        name = risk["name"] if risk else "UNKNOWN"
        print(f"    {risk_id} ({name}): {probes}")
    assert "atlas-jailbreaking" in grouped
    assert "atlas-spreading-toxicity" in grouped
    print("[PASS] Batch grouping works")

    # ============================================================
    # TEST 10: Mitigations (two-hop structure)
    # ============================================================
    section("TEST 10: Mitigations - Two-Hop Structure")
    mit = tm.get_mitigations("atlas-spreading-toxicity")
    print(f"  Mitigation type: {mit['mitigation_type']}")
    print(f"  Available detectors: {len(mit['detectors'])}")
    for d in mit["detectors"]:
        print(f"    {d['id']} ({d['model_name']}) - {d['deployment_status']}")
    assert mit["mitigation_type"] == "content_filtering"
    assert len(mit["detectors"]) >= 2  # HF active + Granite inactive + NeMo built-ins
    print("[PASS] Two-hop structure works")

    # ============================================================
    # TEST 11: Active detectors
    # ============================================================
    section("TEST 11: Active Detectors per Dimension")
    for dim in tm.get_all_dimensions():
        active = tm.get_active_detectors(dim["primary_risk_id"])
        status = f"{len(active)} active" if active else "[FAIL] NO DETECTOR"
        names = [d["id"] for d in active]
        print(f"  {dim['name']:>30}: {status} {names}")
    print("[PASS] Active detector filtering works")

    # ============================================================
    # TEST 12: Coverage gaps
    # ============================================================
    section("TEST 12: Coverage Gaps (tiered)")
    gaps = tm.get_coverage_gaps()
    print(f"  Non-full coverage: {len(gaps)}")
    assert len(gaps) == 2  # hallucination (baseline) + bias (gap)
    for gap in gaps:
        severity = gap["gap_analysis"]["severity"] if gap.get("gap_analysis") else "unknown"
        tier = gap.get("coverage_tier", "unknown")
        print(f"    {gap['dimension_name']} (tier: {tier}, severity: {severity})")
    # Verify tiers
    tier_map = {g["dimension_id"]: g["coverage_tier"] for g in gaps}
    assert tier_map.get("hallucination") == "baseline", f"Expected hallucination=baseline, got {tier_map.get('hallucination')}"
    assert tier_map.get("bias_fairness") == "gap", f"Expected bias_fairness=gap, got {tier_map.get('bias_fairness')}"
    print("[PASS] Tiered gap detection works")

    # ============================================================
    # TEST 13: Coverage summary
    # ============================================================
    section("TEST 13: Coverage Summary (tiered)")
    summary = tm.get_coverage_summary()
    print(f"  Total dimensions: {summary['total_dimensions']}")
    print(f"  Full coverage:    {summary['full']}")
    print(f"  Baseline:         {summary['baseline']}")
    print(f"  Gaps:             {summary['gaps']}")
    print(f"  Coverage:         {summary['coverage_pct']}%")
    assert summary["total_dimensions"] == 5
    assert summary["full"] == 3  # harmful_content, jailbreak, pii_leakage
    assert summary["baseline"] == 1  # hallucination (NeMo built-in available)
    assert summary["gaps"] == 1  # bias_fairness (no detection)
    assert summary["covered"] == 4  # full + baseline
    assert summary["coverage_pct"] == 80.0
    for d in summary["details"]:
        tier = d.get("coverage_tier", "?")
        icon = {"full": "[PASS]", "baseline": "[WARN]", "gap": "[FAIL]"}.get(tier, "?")
        print(f"  {icon} {d['dimension_name']:>30} | tier={tier} | active={d['detector_names']} | all={d['all_detector_names']}")
    print("[PASS] Coverage summary correct")

    # ============================================================
    # TEST 14: Evaluations
    # ============================================================
    section("TEST 14: Evaluations")
    evals = tm.get_evaluations("atlas-hallucination")
    print(f"  Hallucination evaluations: {len(evals)}")
    for e in evals:
        print(f"    [{e['provider']:>25}] {e['task_name']}")
    assert len(evals) == 2
    print("[PASS] Evaluations work")

    # ============================================================
    # TEST 15: Full risk report
    # ============================================================
    section("TEST 15: Full Risk Report - Toxicity")
    report = tm.get_risk_report("atlas-spreading-toxicity")
    print(f"  Risk:       {report['risk']['name']}")
    print(f"  Coverage:   {report['coverage_status']}")
    print(f"  Probes:     {len(report['garak_probes'])}")
    print(f"  Detectors:  {len(report['active_detectors'])}")
    print(f"  Evals:      {len(report['evaluations'])}")
    assert report["coverage_status"] == "full"
    print("[PASS] Risk report works")

    # ============================================================
    # TEST 16: Full dimension report (THE MAIN DEMO QUERY)
    # ============================================================
    section("TEST 16: Dimension Report - harmful_content")
    dreport = tm.get_dimension_report("harmful_content")
    print(f"  Dimension:    {dreport['dimension']['name']}")
    print(f"  Primary risk: {dreport['primary_risk']['name']}")
    print(f"  Coverage:     {dreport['coverage_status']}")
    print(f"  Probes:       {len(dreport['garak_probes'])}")
    print(f"  Active det:   {len(dreport['active_detectors'])}")
    print(f"  All det:      {len(dreport['all_detectors'])}")
    print(f"  Evals:        {len(dreport['evaluations'])}")
    print(f"  Related:      {len(dreport['related_risks'])} risks across taxonomies")
    assert dreport["coverage_status"] == "full"
    assert len(dreport["related_risks"]) == 8

    print("\n  Cross-taxonomy mappings:")
    for mtype, risks in dreport["cross_taxonomy"].items():
        if risks:
            print(f"    {mtype}: {[r['name'] for r in risks]}")

    print("\n  What to scan:")
    for p in dreport["garak_probes"]:
        print(f"    garak --probes {p['probe']}")

    print("\n  What to deploy:")
    for d in dreport["active_detectors"]:
        print(f"    {d['id']}: {d['inference_endpoint'][:55]}...")

    print("\n  What to measure:")
    for e in dreport["evaluations"]:
        print(f"    {e['provider']}: {e['task_name']}")
    print("[PASS] Dimension report works")

    # ============================================================
    # TEST 17: Dimension report for a gap
    # ============================================================
    section("TEST 17: Dimension Report - bias_fairness (GAP)")
    dreport = tm.get_dimension_report("bias_fairness")
    print(f"  Dimension:    {dreport['dimension']['name']}")
    print(f"  Coverage:     {dreport['coverage_status']}")
    print(f"  Detectors:    {len(dreport['active_detectors'])}")
    print(f"  Probes:       {len(dreport['garak_probes'])} (can scan, just can't mitigate)")
    assert dreport["coverage_status"] == "gap"
    assert len(dreport["active_detectors"]) == 0
    assert len(dreport["garak_probes"]) > 0  # We have probes even without detector

    gap = dreport["mitigations"].get("gap_analysis", {})
    print(f"  Gap severity: {gap.get('severity')}")
    print("[PASS] Gap dimension report works")

    # ============================================================
    # TEST 18: Edge cases
    # ============================================================
    section("TEST 18: Edge Cases")
    assert tm.get_risk("nonexistent-risk") is None
    print("  [PASS] Unknown risk returns None")
    assert tm.get_dimension_report("nonexistent-dim") is None
    print("  [PASS] Unknown dimension returns None")
    assert tm.get_risk_report("nonexistent-risk") is None
    print("  [PASS] Unknown risk report returns None")
    assert tm.get_garak_probes("atlas-data-poisoning") == []
    print("  [PASS] Risk without operational data returns empty list")
    assert tm.get_dimension_for_risk("atlas-data-poisoning") is None
    print("  [PASS] Risk not in any dimension returns None")

    # ============================================================
    # TEST 19: resolve_profile -- base defaults (no domain)
    # ============================================================
    section("TEST 19: Profile -- Base Defaults")
    profile = tm.resolve_profile()
    assert profile["profile_id"] == "base"
    assert profile["domain"] is None
    assert profile["use_case"] is None
    assert profile["thresholds"]["jailbreak"] == {"fail": 0.80, "warn": 0.95}
    assert profile["thresholds"]["hallucination"] == {"fail": 0.50, "warn": 0.80}
    assert profile["dimension_priorities"]["jailbreak"] == "high"
    assert profile["additional_evals"] == []
    print(f"  Profile: {profile['profile_id']}")
    print(f"  Thresholds: {len(profile['thresholds'])} dimensions")
    print("[PASS] Base defaults correct")

    # ============================================================
    # TEST 20: resolve_profile -- domain overrides
    # ============================================================
    section("TEST 20: Profile -- Domain Overrides (healthcare)")
    profile = tm.resolve_profile(domain_id="healthcare")
    assert profile["profile_id"] == "healthcare"
    assert profile["domain"] == "Healthcare"
    # Healthcare overrides hallucination to stricter
    assert profile["thresholds"]["hallucination"] == {"fail": 0.90, "warn": 0.98}
    assert profile["thresholds"]["bias_fairness"] == {"fail": 0.80, "warn": 0.95}
    # Non-overridden fields inherit from base
    assert profile["thresholds"]["jailbreak"] == {"fail": 0.80, "warn": 0.95}
    assert profile["dimension_priorities"]["hallucination"] == "critical"
    print(f"  Profile: {profile['profile_id']}")
    print(f"  Hallucination threshold: {profile['thresholds']['hallucination']}")
    print("[PASS] Domain overrides merge correctly")

    # ============================================================
    # TEST 21: resolve_profile -- use-case overrides on top of domain
    # ============================================================
    section("TEST 21: Profile -- Use-Case Overrides (healthcare/billing_coding)")
    profile = tm.resolve_profile(domain_id="healthcare", use_case_id="billing_coding")
    assert profile["profile_id"] == "healthcare/billing_coding"
    assert profile["use_case"] == "Medical Billing & Coding"
    # Use-case overrides PII back down from domain's inherited value
    assert profile["thresholds"]["pii_leakage"] == {"fail": 0.60, "warn": 0.80}
    # Use-case overrides hallucination further from domain
    assert profile["thresholds"]["hallucination"] == {"fail": 0.95, "warn": 0.99}
    # PII priority overridden back to medium
    assert profile["dimension_priorities"]["pii_leakage"] == "medium"
    print(f"  Profile: {profile['profile_id']}")
    print(f"  PII threshold: {profile['thresholds']['pii_leakage']}")
    print(f"  Hallucination threshold: {profile['thresholds']['hallucination']}")
    print("[PASS] Use-case overrides on top of domain correct")

    # ============================================================
    # TEST 22: resolve_profile -- inheritance (unoverridden fields)
    # ============================================================
    section("TEST 22: Profile -- Inheritance (unoverridden fields)")
    profile = tm.resolve_profile(domain_id="healthcare", use_case_id="patient_chatbot")
    # harmful_content not overridden at any level -- should be base
    assert profile["thresholds"]["harmful_content"] == {"fail": 0.50, "warn": 0.80}
    # jailbreak not overridden -- should be base
    assert profile["dimension_priorities"]["jailbreak"] == "high"
    # PII overridden at use-case level
    assert profile["thresholds"]["pii_leakage"] == {"fail": 0.90, "warn": 0.98}
    # Hallucination from domain level (not overridden at use-case)
    assert profile["thresholds"]["hallucination"] == {"fail": 0.90, "warn": 0.98}
    print(f"  harmful_content threshold (base): {profile['thresholds']['harmful_content']}")
    print(f"  hallucination threshold (domain): {profile['thresholds']['hallucination']}")
    print(f"  pii_leakage threshold (use-case): {profile['thresholds']['pii_leakage']}")
    print("[PASS] Inheritance chain correct")

    # ============================================================
    # TEST 23: resolve_profile -- unknown domain raises ValueError
    # ============================================================
    section("TEST 23: Profile -- Unknown Domain Error")
    try:
        tm.resolve_profile(domain_id="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Raised ValueError: {e}")
    print("[PASS] Unknown domain raises ValueError")

    # ============================================================
    # TEST 24: resolve_profile -- use_case without domain raises ValueError
    # ============================================================
    section("TEST 24: Profile -- Use-Case Without Domain Error")
    try:
        tm.resolve_profile(use_case_id="patient_chatbot")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Raised ValueError: {e}")
    print("[PASS] Use-case without domain raises ValueError")

    # ============================================================
    # TEST 25: generate_collection
    # ============================================================
    section("TEST 25: Generate Collection (healthcare)")
    collection = tm.generate_collection(domain_id="healthcare")
    assert "profile" in collection
    assert "dimensions" in collection
    assert len(collection["dimensions"]) == 5
    for dim_id, dim_data in collection["dimensions"].items():
        assert "priority" in dim_data
        assert "thresholds" in dim_data
        assert "probes" in dim_data
        assert "evaluations" in dim_data
        assert "detectors" in dim_data
        assert "coverage_status" in dim_data
    # Healthcare hallucination should be critical priority
    assert collection["dimensions"]["hallucination"]["priority"] == "critical"
    assert collection["dimensions"]["hallucination"]["coverage_status"] == "baseline"
    assert collection["dimensions"]["harmful_content"]["coverage_status"] == "full"
    print(f"  Dimensions: {list(collection['dimensions'].keys())}")
    print(f"  Hallucination priority: {collection['dimensions']['hallucination']['priority']}")
    print("[PASS] Collection structure correct")

    # ============================================================
    # TEST 26: generate_collection -- additional_evals appended
    # ============================================================
    section("TEST 26: Collection -- Additional Evals (patient_chatbot)")
    collection = tm.generate_collection(domain_id="healthcare", use_case_id="patient_chatbot")
    hall_evals = collection["dimensions"]["hallucination"]["evaluations"]
    eval_ids = [e["id"] for e in hall_evals]
    assert "medqa" in eval_ids, f"Expected medqa in evals, got {eval_ids}"
    # Base evals should still be there
    assert "truthfulqa" in eval_ids
    print(f"  Hallucination evals: {eval_ids}")
    print("[PASS] Additional evals appended correctly")

    # ============================================================
    # TEST 27: get_all_domains
    # ============================================================
    section("TEST 27: Get All Domains")
    domains = tm.get_all_domains()
    assert len(domains) == 3
    domain_ids = [d["id"] for d in domains]
    assert "healthcare" in domain_ids
    assert "finance" in domain_ids
    assert "general" in domain_ids
    # Healthcare should have 3 use-cases
    hc = [d for d in domains if d["id"] == "healthcare"][0]
    assert len(hc["use_cases"]) == 3
    print(f"  Domains: {domain_ids}")
    print(f"  Healthcare use-cases: {[uc['id'] for uc in hc['use_cases']]}")
    print("[PASS] Domain listing correct")

    # ============================================================
    # TEST 28: dimension report with profile
    # ============================================================
    section("TEST 28: Dimension Report with Profile")
    dreport = tm.get_dimension_report("hallucination", domain_id="healthcare",
                                       use_case_id="patient_chatbot")
    assert "profile_thresholds" in dreport
    assert "profile_priority" in dreport
    assert dreport["profile_thresholds"] == {"fail": 0.90, "warn": 0.98}
    assert dreport["profile_priority"] == "critical"
    # Should include additional eval (medqa)
    eval_ids = [e["id"] for e in dreport["evaluations"]]
    assert "medqa" in eval_ids
    print(f"  Profile thresholds: {dreport['profile_thresholds']}")
    print(f"  Profile priority: {dreport['profile_priority']}")
    print(f"  Evaluations: {eval_ids}")
    print("[PASS] Profile-aware dimension report correct")

    # ============================================================
    # TEST 29: backward compatibility
    # ============================================================
    section("TEST 29: Backward Compatibility")
    # Dimension report without profile should NOT have profile keys
    dreport = tm.get_dimension_report("harmful_content")
    assert "profile_thresholds" not in dreport
    assert "profile_priority" not in dreport
    assert dreport["coverage_status"] == "full"
    assert len(dreport["related_risks"]) == 8
    print("  Dimension report without profile: no profile keys present")
    print("[PASS] Backward compatibility maintained")

    # ============================================================
    # TEST 30: compliance coverage -- OWASP
    # ============================================================
    section("TEST 30: Compliance Coverage (OWASP)")
    owasp_cov = tm.get_compliance_coverage("owasp-llm-2.0")
    assert owasp_cov["total_risks"] == 10
    assert owasp_cov["mapped"] >= 3  # LLM01->jailbreak, LLM02->pii, LLM09->hallucination
    assert "jailbreak" in owasp_cov["dimensions_touched"]
    assert "pii_leakage" in owasp_cov["dimensions_touched"]
    # LLM01 should map to jailbreak with full coverage
    llm01 = [d for d in owasp_cov["details"] if "Prompt Injection" in d["risk_name"]][0]
    assert llm01["mapped_dimension"] == "jailbreak"
    assert llm01["coverage_tier"] == "full"
    print(f"  OWASP: {owasp_cov['mapped']}/{owasp_cov['total_risks']} risks mapped to {len(owasp_cov['dimensions_touched'])} dimensions")
    print(f"  Full: {owasp_cov['full']}, Baseline: {owasp_cov['baseline']}, Gaps: {owasp_cov['gaps']}")
    print("[PASS] OWASP compliance coverage correct")

    # ============================================================
    # TEST 31: compliance coverage -- NIST
    # ============================================================
    section("TEST 31: Compliance Coverage (NIST)")
    nist_cov = tm.get_compliance_coverage("nist-ai-rmf")
    assert nist_cov["total_risks"] == 12
    assert nist_cov["mapped"] >= 9
    assert "hallucination" in nist_cov["dimensions_touched"]
    assert "bias_fairness" in nist_cov["dimensions_touched"]
    # Confabulation should map to hallucination
    confab = [d for d in nist_cov["details"] if "Confabulation" in d["risk_name"]][0]
    assert confab["mapped_dimension"] == "hallucination"
    assert confab["coverage_tier"] == "baseline"
    print(f"  NIST: {nist_cov['mapped']}/{nist_cov['total_risks']} risks mapped to {len(nist_cov['dimensions_touched'])} dimensions")
    print(f"  Full: {nist_cov['full']}, Baseline: {nist_cov['baseline']}, Gaps: {nist_cov['gaps']}")
    print("[PASS] NIST compliance coverage correct")

    # ============================================================
    # TEST 32: compliance coverage -- unknown taxonomy
    # ============================================================
    section("TEST 32: Compliance Coverage (Unknown Taxonomy)")
    try:
        tm.get_compliance_coverage("nonexistent-standard")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ValueError raised for unknown taxonomy")
    print("[PASS] Unknown taxonomy handled correctly")

    # ============================================================
    # DONE
    # ============================================================
    section("ALL 32 TESTS PASSED [PASS]")
    print(f"Taxonomy: {stats['total_risks']} risks across {len(stats['taxonomy_counts'])} taxonomies")
    print(f"Operational: {stats['operational_dimensions']} dimensions, {stats['total_probes_mapped']} probes mapped")
    print(f"Coverage: {summary['covered']}/{summary['total_dimensions']} dimensions protected ({summary['full']} full, {summary['baseline']} baseline, {summary['gaps']} gap)")


if __name__ == "__main__":
    main()
