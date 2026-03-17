"""
TrustyAI Risk Taxonomy - Query Layer

Loads the taxonomy YAML files and provides a query interface.

Usage:
    from taxonomy.query import TaxonomyManager

    tm = TaxonomyManager()

    # Get everything about a dimension (the main demo query)
    report = tm.get_dimension_report("harmful_content")

    # Get everything about a specific risk
    report = tm.get_risk_report("atlas-spreading-toxicity")

    # Reverse lookup: Garak found a vulnerability, what risk is it?
    risk_id = tm.lookup_risk_by_probe("dan.DanInTheWild")

    # Coverage analysis
    gaps = tm.get_coverage_gaps()
    summary = tm.get_coverage_summary()
"""

import copy
import os
import yaml
from pathlib import Path
from typing import Optional


def _merge_overrides(base: dict, overrides: dict) -> dict:
    """Merge profile overrides into a base profile.

    - 'thresholds' and 'dimension_priorities': shallow dict merge per key.
    - 'additional_evals': appended (accumulated from all levels).
    - 'guardrail_overrides': shallow dict merge per key (future use).
    """
    if not overrides:
        return base

    result = copy.deepcopy(base)
    for key in ("thresholds", "dimension_priorities", "guardrail_overrides"):
        if key in overrides:
            if key not in result:
                result[key] = {}
            result[key].update(overrides[key])

    if "additional_evals" in overrides:
        if "additional_evals" not in result:
            result["additional_evals"] = []
        result["additional_evals"].extend(overrides["additional_evals"])

    return result


class TaxonomyManager:
    """
    Loads and queries the TrustyAI risk taxonomy.

    Data files:
        risk_taxonomy.yaml       - 546 risks from Atlas Nexus (reference layer)
        optimizer_dimensions.yaml - 5 operational risk dimensions
        risk_to_garak.yaml       - risk -> Garak probe mappings
        risk_to_mitigations.yaml - risk -> detector mappings (two-hop)
        risk_to_eval.yaml        - risk -> evaluation benchmark mappings
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Load all YAML data files and build indexes.

        Args:
            data_dir: Path to the taxonomy/data/ directory.
                      Defaults to the data/ dir relative to this file.
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")

        self._data_dir = Path(data_dir)

        # Load data files
        self._taxonomy = self._load("risk_taxonomy.yaml")
        self._dimensions = self._load("optimizer_dimensions.yaml")
        self._garak = self._load("risk_to_garak.yaml")
        self._mitigations = self._load("risk_to_mitigations.yaml")
        self._evals = self._load("risk_to_eval.yaml")

        # Build indexes
        self._risk_index = {r["id"]: r for r in self._taxonomy["risks"]}
        self._dimension_index = {
            d["id"]: d for d in self._dimensions["dimensions"]
        }
        self._garak_index = {
            m["risk_id"]: m for m in self._garak["risk_garak_mappings"]
        }
        self._mitigation_index = {
            m["risk_id"]: m for m in self._mitigations["risk_mitigation_mappings"]
        }
        self._eval_index = {
            m["risk_id"]: m for m in self._evals["risk_evaluation_mappings"]
        }

        # Load domain profiles
        self._profiles = self._load("domain_profiles.yaml")
        self._domain_index = {}
        self._use_case_index = {}
        for domain in self._profiles.get("domains", []):
            self._domain_index[domain["id"]] = domain
            for uc in domain.get("use_cases", []):
                self._use_case_index[(domain["id"], uc["id"])] = uc

        # Reverse indexes
        self._probe_to_risk = {}
        for mapping in self._garak["risk_garak_mappings"]:
            for probe in mapping["probes"]:
                self._probe_to_risk[probe["probe"]] = mapping["risk_id"]

        self._risk_to_dimension = {}
        self._risk_to_dimension_all = {}  # includes related_risk_ids
        for dim in self._dimensions["dimensions"]:
            if dim["primary_risk_id"]:
                self._risk_to_dimension[dim["primary_risk_id"]] = dim["id"]
                self._risk_to_dimension_all[dim["primary_risk_id"]] = dim["id"]
            for rid in dim.get("related_risk_ids", []):
                self._risk_to_dimension_all[rid] = dim["id"]

    def _load(self, filename: str) -> dict:
        """Load a YAML file from the data directory."""
        filepath = self._data_dir / filename
        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    # ================================================================
    # Risk Queries (546 risks from Atlas Nexus)
    # ================================================================

    def get_risk(self, risk_id: str) -> Optional[dict]:
        """
        Get risk definition by ID.

        Args:
            risk_id: Atlas Nexus risk ID (e.g., "atlas-spreading-toxicity")

        Returns:
            Risk dict or None if not found.
        """
        return self._risk_index.get(risk_id)

    def get_all_risks(self) -> list[dict]:
        """Get all 546 risks in the taxonomy."""
        return self._taxonomy["risks"]

    def get_risks_by_taxonomy(self, taxonomy: str) -> list[dict]:
        """
        Get all risks from a specific source taxonomy.

        Args:
            taxonomy: e.g., "ibm-risk-atlas", "nist-ai-rmf", "owasp-llm-2.0"
        """
        return [
            r for r in self._taxonomy["risks"]
            if r.get("isDefinedByTaxonomy") == taxonomy
        ]

    def get_related_risks(self, risk_id: str) -> dict:
        """
        Get all cross-taxonomy mappings for a risk.

        Returns dict with broad_mappings, related_mappings, exact_mappings,
        narrow_mappings, close_mappings -- each resolved to full risk dicts
        where possible.
        """
        risk = self._risk_index.get(risk_id)
        if not risk:
            return {}

        result = {}
        for mapping_type in ["broad_mappings", "related_mappings", "exact_mappings",
                             "narrow_mappings", "close_mappings"]:
            mapped_ids = risk.get(mapping_type, []) or []
            resolved = []
            for mid in mapped_ids:
                mapped_risk = self._risk_index.get(mid)
                if mapped_risk:
                    resolved.append(mapped_risk)
                else:
                    resolved.append({"id": mid, "name": mid, "description": None})
            result[mapping_type] = resolved
        return result

    # ================================================================
    # Dimension Queries (5 operational dimensions)
    # ================================================================

    def get_dimension(self, dimension_id: str) -> Optional[dict]:
        """
        Get an operational dimension by ID.

        Args:
            dimension_id: e.g., "harmful_content", "jailbreak", "pii_leakage"
        """
        return self._dimension_index.get(dimension_id)

    def get_all_dimensions(self) -> list[dict]:
        """Get all operational dimensions."""
        return self._dimensions["dimensions"]

    def get_dimension_for_risk(self, risk_id: str) -> Optional[str]:
        """
        Find which dimension a risk belongs to (if any).

        Returns dimension_id or None.
        """
        return self._risk_to_dimension.get(risk_id)

    # ================================================================
    # Operational Queries (probes, detectors, evals)
    # ================================================================

    def get_garak_probes(self, risk_id: str, min_priority: str = "LOW") -> list[dict]:
        """
        Get Garak probes for a risk, optionally filtered by priority.

        Args:
            risk_id: Atlas Nexus risk ID
            min_priority: "HIGH", "MEDIUM", or "LOW"
        """
        mapping = self._garak_index.get(risk_id)
        if not mapping:
            return []

        priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_level = priority_order.get(min_priority, 1)

        return [
            p for p in mapping["probes"]
            if priority_order.get(p["priority"], 0) >= min_level
        ]

    def get_mitigations(self, risk_id: str) -> Optional[dict]:
        """
        Get mitigation info for a risk (two-hop structure).

        Returns dict with mitigation_type, detectors, and gap_analysis.
        """
        return self._mitigation_index.get(risk_id)

    def get_active_detectors(self, risk_id: str) -> list[dict]:
        """Get only active (deployed) detectors for a risk."""
        mitigation = self._mitigation_index.get(risk_id)
        if not mitigation:
            return []
        return [
            d for d in mitigation.get("detectors", [])
            if d.get("deployment_status") == "active"
        ]

    def get_evaluations(self, risk_id: str) -> list[dict]:
        """Get evaluation benchmarks for a risk."""
        mapping = self._eval_index.get(risk_id)
        if not mapping:
            return []
        return mapping.get("evaluations", [])

    # ================================================================
    # Reverse Lookups
    # ================================================================

    def lookup_risk_by_probe(self, probe_name: str) -> Optional[str]:
        """
        Reverse lookup: Garak probe name -> risk ID.

        Args:
            probe_name: Exact Garak probe string (e.g., "dan.DanInTheWild")
        """
        return self._probe_to_risk.get(probe_name)

    def lookup_risks_by_probes(self, probe_names: list[str]) -> dict[str, list[str]]:
        """
        Batch reverse lookup. Groups probes by risk.

        Returns dict of risk_id -> list of probe names.
        Unmapped probes go under "unknown".
        """
        result = {}
        for probe in probe_names:
            risk_id = self._probe_to_risk.get(probe, "unknown")
            if risk_id not in result:
                result[risk_id] = []
            result[risk_id].append(probe)
        return result

    # ================================================================
    # Coverage Analysis (operates on dimensions, not all 546 risks)
    # ================================================================

    def _get_coverage_tier(self, risk_id: str) -> str:
        """
        Determine coverage tier for a risk.

        Returns:
            "full" -- has active detector with measured effectiveness
            "baseline" -- has available (but not active/measured) detectors
            "gap" -- no detectors at all
        """
        mitigation = self._mitigation_index.get(risk_id)
        if not mitigation:
            return "gap"
        detectors = mitigation.get("detectors", [])
        if not detectors:
            return "gap"
        if any(d.get("deployment_status") == "active" for d in detectors):
            return "full"
        if any(d.get("deployment_status") == "available" for d in detectors):
            return "baseline"
        return "gap"

    def get_coverage_gaps(self) -> list[dict]:
        """
        Find dimensions without full coverage (baseline or gap).

        Returns list of dicts with dimension, gap_analysis, and coverage_tier.
        """
        gaps = []
        for dim in self._dimensions["dimensions"]:
            risk_id = dim["primary_risk_id"]
            tier = self._get_coverage_tier(risk_id) if risk_id else "gap"
            if tier != "full":
                mitigation = self._mitigation_index.get(risk_id, {})
                gaps.append({
                    "dimension_id": dim["id"],
                    "dimension_name": dim["name"],
                    "category": dim["category"],
                    "primary_risk_id": risk_id,
                    "coverage_tier": tier,
                    "gap_analysis": mitigation.get("gap_analysis"),
                })
        return gaps

    def get_all_detectors(self, risk_id: str) -> list[dict]:
        """Get all detectors for a risk (active, available, and inactive)."""
        mitigation = self._mitigation_index.get(risk_id)
        if not mitigation:
            return []
        return mitigation.get("detectors", [])

    def get_coverage_summary(self) -> dict:
        """
        Coverage summary across all operational dimensions.

        Returns dict with total, covered (full+baseline), gaps,
        coverage_pct, and per-dimension details with coverage_tier.
        """
        details = []
        full = 0
        baseline = 0
        for dim in self._dimensions["dimensions"]:
            risk_id = dim["primary_risk_id"]
            tier = self._get_coverage_tier(risk_id) if risk_id else "gap"
            active = self.get_active_detectors(risk_id) if risk_id else []
            all_detectors = self.get_all_detectors(risk_id) if risk_id else []
            if tier == "full":
                full += 1
            elif tier == "baseline":
                baseline += 1
            details.append({
                "dimension_id": dim["id"],
                "dimension_name": dim["name"],
                "category": dim["category"],
                "primary_risk_id": risk_id,
                "num_active_detectors": len(active),
                "num_available_detectors": len(all_detectors),
                "detector_names": [d["id"] for d in active],
                "all_detector_names": [d["id"] for d in all_detectors],
                "covered": tier != "gap",
                "coverage_tier": tier,
            })

        total = len(self._dimensions["dimensions"])
        covered = full + baseline
        return {
            "total_dimensions": total,
            "covered": covered,
            "full": full,
            "baseline": baseline,
            "gaps": total - covered,
            "coverage_pct": round(covered / total * 100, 1) if total > 0 else 0,
            "details": details,
        }

    def get_compliance_coverage(self, taxonomy_id: str) -> dict:
        """
        Trace a compliance standard's risks through cross-taxonomy mappings
        to our operational dimensions and return coverage status.

        Args:
            taxonomy_id: Source taxonomy ID (e.g., "owasp-llm-2.0", "nist-ai-rmf")

        Returns:
            Dict with standard info, per-risk tracing results, and summary counts.
        """
        standard_risks = self.get_risks_by_taxonomy(taxonomy_id)
        if not standard_risks:
            raise ValueError(f"Unknown taxonomy: '{taxonomy_id}'")

        details = []
        mapped_dims = set()

        for risk in standard_risks:
            rid = risk["id"]
            trace = {
                "risk_id": rid,
                "risk_name": risk["name"],
                "mapped_dimension": None,
                "mapping_path": None,
                "coverage_tier": None,
                "effectiveness": None,
            }

            # Direct match: is this risk a primary or related risk of a dimension?
            dim_id = self._risk_to_dimension_all.get(rid)
            if dim_id:
                trace["mapped_dimension"] = dim_id
                trace["mapping_path"] = "direct"
            else:
                # Cross-taxonomy trace: follow mappings to find a dimension
                related = self.get_related_risks(rid)
                for mtype in ["exact_mappings", "broad_mappings", "close_mappings",
                              "narrow_mappings", "related_mappings"]:
                    if trace["mapped_dimension"]:
                        break
                    for mapped_risk in related.get(mtype, []):
                        target_dim = self._risk_to_dimension_all.get(mapped_risk["id"])
                        if target_dim:
                            trace["mapped_dimension"] = target_dim
                            trace["mapping_path"] = f"{mtype} -> {mapped_risk['id']}"
                            break

            # Get coverage for the mapped dimension
            if trace["mapped_dimension"]:
                dim = self._dimension_index.get(trace["mapped_dimension"])
                if dim:
                    primary_rid = dim["primary_risk_id"]
                    trace["coverage_tier"] = self._get_coverage_tier(primary_rid)
                    active = self.get_active_detectors(primary_rid)
                    if active:
                        eff = active[0].get("effectiveness_score")
                        trace["effectiveness"] = eff
                    mapped_dims.add(trace["mapped_dimension"])

            details.append(trace)

        mapped = [d for d in details if d["mapped_dimension"]]
        full = sum(1 for d in mapped if d["coverage_tier"] == "full")
        baseline = sum(1 for d in mapped if d["coverage_tier"] == "baseline")
        gaps = sum(1 for d in mapped if d["coverage_tier"] == "gap")
        unmapped = len(details) - len(mapped)

        return {
            "taxonomy_id": taxonomy_id,
            "total_risks": len(details),
            "mapped": len(mapped),
            "unmapped": unmapped,
            "full": full,
            "baseline": baseline,
            "gaps": gaps,
            "dimensions_touched": sorted(mapped_dims),
            "details": details,
        }

    # ================================================================
    # Domain & Use-Case Profiles
    # ================================================================

    def resolve_profile(self, domain_id: Optional[str] = None, use_case_id: Optional[str] = None) -> dict:
        """
        Resolve a fully-merged profile from base_defaults + domain + use_case.

        Args:
            domain_id: Domain profile ID (e.g., "healthcare").
            use_case_id: Use-case ID within a domain (e.g., "billing_coding").
                         Requires domain_id.

        Returns:
            Fully resolved profile dict with keys: profile_id, domain, use_case,
            thresholds, dimension_priorities, additional_evals, guardrail_overrides.

        Raises:
            ValueError: Unknown domain_id, or use_case_id without domain_id.
        """
        if use_case_id and not domain_id:
            raise ValueError("use_case_id requires domain_id")

        base = copy.deepcopy(self._profiles["base_defaults"])
        base.setdefault("additional_evals", [])
        base.setdefault("guardrail_overrides", {})

        profile_id = "base"
        domain_name = None
        use_case_name = None

        if domain_id:
            domain = self._domain_index.get(domain_id)
            if not domain:
                raise ValueError(f"Unknown domain: '{domain_id}'")
            base = _merge_overrides(base, domain.get("overrides", {}))
            profile_id = domain_id
            domain_name = domain["name"]

            if use_case_id:
                uc = self._use_case_index.get((domain_id, use_case_id))
                if not uc:
                    raise ValueError(
                        f"Unknown use-case '{use_case_id}' in domain '{domain_id}'"
                    )
                base = _merge_overrides(base, uc.get("overrides", {}))
                profile_id = f"{domain_id}/{use_case_id}"
                use_case_name = uc["name"]

        return {
            "profile_id": profile_id,
            "domain": domain_name,
            "use_case": use_case_name,
            "thresholds": base["thresholds"],
            "dimension_priorities": base["dimension_priorities"],
            "additional_evals": base["additional_evals"],
            "guardrail_overrides": base["guardrail_overrides"],
        }

    def generate_collection(self, domain_id: Optional[str] = None, use_case_id: Optional[str] = None) -> dict:
        """
        Generate a collection recommendation for a domain/use-case.

        Resolves the profile, then for each of the 5 dimensions produces:
        priority, thresholds, probes, evaluations, detectors, coverage_status.

        Returns dict with profile info and per-dimension collection data.
        """
        profile = self.resolve_profile(domain_id, use_case_id)
        dimensions = {}

        for dim in self._dimensions["dimensions"]:
            dim_id = dim["id"]
            risk_id = dim["primary_risk_id"]

            # Probes
            probes = self.get_garak_probes(risk_id) if risk_id else []

            # Evaluations: base + additional from profile
            evals = list(self.get_evaluations(risk_id)) if risk_id else []
            for ae in profile["additional_evals"]:
                if ae.get("applies_to_dimension") == dim_id:
                    evals.append(ae)

            # Detectors (all options, not just active)
            active = self.get_active_detectors(risk_id) if risk_id else []
            all_detectors = self.get_all_detectors(risk_id) if risk_id else []
            tier = self._get_coverage_tier(risk_id) if risk_id else "gap"

            dimensions[dim_id] = {
                "dimension_name": dim["name"],
                "priority": profile["dimension_priorities"].get(dim_id, "medium"),
                "thresholds": profile["thresholds"].get(dim_id, {}),
                "probes": probes,
                "evaluations": evals,
                "detectors": active,
                "all_detectors": all_detectors,
                "coverage_status": tier,
            }

        return {
            "profile": profile,
            "dimensions": dimensions,
        }

    def get_all_domains(self) -> list[dict]:
        """List available domains with their use-cases."""
        result = []
        for domain in self._profiles.get("domains", []):
            use_cases = [
                {"id": uc["id"], "name": uc["name"]}
                for uc in domain.get("use_cases", [])
            ]
            result.append({
                "id": domain["id"],
                "name": domain["name"],
                "description": domain.get("description", ""),
                "use_cases": use_cases,
            })
        return result

    def get_thresholds(self, domain_id: Optional[str] = None, use_case_id: Optional[str] = None) -> dict:
        """Convenience method: resolve profile and return just the thresholds."""
        profile = self.resolve_profile(domain_id, use_case_id)
        return profile["thresholds"]

    # ================================================================
    # Full Reports
    # ================================================================

    def get_risk_report(self, risk_id: str) -> Optional[dict]:
        """
        Get everything the taxonomy knows about a specific risk.

        Returns risk definition, probes, mitigations, evaluations,
        active detectors, and coverage status.
        """
        risk = self.get_risk(risk_id)
        if not risk:
            return None

        active = self.get_active_detectors(risk_id)
        tier = self._get_coverage_tier(risk_id)

        return {
            "risk": risk,
            "garak_probes": self.get_garak_probes(risk_id),
            "mitigations": self.get_mitigations(risk_id),
            "evaluations": self.get_evaluations(risk_id),
            "active_detectors": active,
            "all_detectors": self.get_all_detectors(risk_id),
            "coverage_status": tier,
        }

    def get_dimension_report(self, dimension_id: str,
                             domain_id: Optional[str] = None,
                             use_case_id: Optional[str] = None) -> Optional[dict]:
        """
        Get everything about an operational dimension -- the main demo query.

        Combines dimension info + primary risk details + all operational data
        + related risks from across taxonomies. When domain/use_case provided,
        adds profile-specific thresholds, priorities, and additional evals.

        Args:
            dimension_id: e.g., "harmful_content", "jailbreak"
            domain_id: Optional domain profile ID for profile-aware report.
            use_case_id: Optional use-case ID (requires domain_id).

        Returns complete report or None if dimension not found.
        """
        dim = self._dimension_index.get(dimension_id)
        if not dim:
            return None

        risk_id = dim["primary_risk_id"]
        risk = self.get_risk(risk_id) if risk_id else None
        active = self.get_active_detectors(risk_id) if risk_id else []
        mitigation = self.get_mitigations(risk_id) if risk_id else None

        # Resolve related risks to full definitions
        related = []
        for rid in dim.get("related_risk_ids", []):
            related_risk = self.get_risk(rid)
            if related_risk:
                related.append(related_risk)
            else:
                related.append({"id": rid, "name": rid})

        # Base evaluations
        evals = list(self.get_evaluations(risk_id)) if risk_id else []

        tier = self._get_coverage_tier(risk_id) if risk_id else "gap"
        all_detectors = self.get_all_detectors(risk_id) if risk_id else []

        report = {
            "dimension": dim,
            "primary_risk": risk,
            "cross_taxonomy": self.get_related_risks(risk_id) if risk_id else {},
            "garak_probes": self.get_garak_probes(risk_id) if risk_id else [],
            "mitigations": mitigation,
            "evaluations": evals,
            "active_detectors": active,
            "all_detectors": all_detectors,
            "coverage_status": tier,
            "related_risks": related,
        }

        # Add profile data when domain/use_case provided
        if domain_id is not None:
            profile = self.resolve_profile(domain_id, use_case_id)
            report["profile_thresholds"] = profile["thresholds"].get(dimension_id, {})
            report["profile_priority"] = profile["dimension_priorities"].get(
                dimension_id, "medium"
            )
            for ae in profile["additional_evals"]:
                if ae.get("applies_to_dimension") == dimension_id:
                    report["evaluations"].append(ae)

        return report

    # ================================================================
    # Write-back (update taxonomy with measured data)
    # ================================================================

    def update_detector_effectiveness(self, detector_id, effectiveness_score,
                                      latency_ms=None):
        """
        Write measured effectiveness back into risk_to_mitigations.yaml.

        Closes the governance feedback loop: scan results improve
        future recommendations.

        Args:
            detector_id: Detector ID (e.g., "jailbreak-detector-hf")
            effectiveness_score: Measured effectiveness (0.0 to 1.0)
            latency_ms: Optional measured latency in milliseconds
        """
        mitigations_path = self._data_dir / "risk_to_mitigations.yaml"

        with open(mitigations_path, "r") as f:
            data = yaml.safe_load(f)

        updated = False
        for mapping in data["risk_mitigation_mappings"]:
            for detector in mapping.get("detectors", []):
                if detector["id"] == detector_id:
                    detector["effectiveness_score"] = round(effectiveness_score, 4)
                    if latency_ms is not None:
                        detector["latency_ms"] = round(latency_ms, 1)
                    updated = True
                    break
            if updated:
                break

        if not updated:
            raise ValueError(f"Detector '{detector_id}' not found in risk_to_mitigations.yaml")

        with open(mitigations_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Reload to rebuild indexes
        self._mitigations = self._load("risk_to_mitigations.yaml")
        self._mitigation_index = {
            m["risk_id"]: m for m in self._mitigations["risk_mitigation_mappings"]
        }

    # ================================================================
    # Taxonomy Metadata
    # ================================================================

    def get_taxonomy_stats(self) -> dict:
        """Get high-level stats about the taxonomy."""
        return {
            "total_risks": len(self._taxonomy["risks"]),
            "taxonomy_counts": self._taxonomy.get("taxonomy_counts", {}),
            "operational_dimensions": len(self._dimensions["dimensions"]),
            "risks_with_garak_probes": len(self._garak_index),
            "risks_with_detectors": len(self._mitigation_index),
            "risks_with_evaluations": len(self._eval_index),
            "total_probes_mapped": sum(
                len(m["probes"]) for m in self._garak["risk_garak_mappings"]
            ),
        }
