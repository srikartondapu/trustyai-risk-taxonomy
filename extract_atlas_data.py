"""
Extract ALL data from Atlas Nexus for TrustyAI Risk Taxonomy.
Extracts ALL attributes dynamically instead of hardcoded field names.

Run this in your trustyai-risk-taxonomy venv:
    python extract_atlas_data.py

Outputs: atlas_nexus_full_export.json
"""

import json
import warnings
from ai_atlas_nexus import AIAtlasNexus

# Suppress pydantic deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fields to skip (internal pydantic/linkml stuff, not useful data)
SKIP_FIELDS = {
    "model_computed_fields", "model_config", "model_extra",
    "model_fields", "model_fields_set", "linkml_meta",
}


def extract_all_fields(obj):
    """Extract ALL non-private, non-callable attributes from an object."""
    result = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        if attr in SKIP_FIELDS:
            continue
        try:
            val = getattr(obj, attr)
            if callable(val):
                continue
            # Convert to JSON-serializable types
            if val is None:
                result[attr] = None
            elif isinstance(val, (str, int, float, bool)):
                result[attr] = val
            elif isinstance(val, list):
                result[attr] = [
                    str(item) if not isinstance(item, (str, int, float, bool, type(None)))
                    else item
                    for item in val
                ]
            else:
                result[attr] = str(val)
        except Exception as e:
            print(f"  Warning: could not serialize attribute '{attr}': {e}")
    return result


def main():
    print("Initializing Atlas Nexus...")
    atlas = AIAtlasNexus()

    # ============================================================
    # Extract ALL risks
    # ============================================================
    print("\nExtracting risks...")
    all_risks = atlas.get_all_risks()
    print(f"  Found {len(all_risks)} risks")

    risks_data = []
    taxonomies_seen = set()

    for risk in all_risks:
        extracted = extract_all_fields(risk)
        risks_data.append(extracted)
        tax = extracted.get("isDefinedByTaxonomy", "")
        if tax:
            taxonomies_seen.add(tax)

    # ============================================================
    # Extract ALL actions
    # ============================================================
    print("\nExtracting actions...")
    all_actions = atlas.get_all_actions()
    print(f"  Found {len(all_actions)} actions")

    actions_data = []
    for action in all_actions:
        actions_data.append(extract_all_fields(action))

    # ============================================================
    # Build stats
    # ============================================================
    taxonomy_counts = {}
    for risk in risks_data:
        tax = risk.get("isDefinedByTaxonomy", "unknown")
        taxonomy_counts[tax] = taxonomy_counts.get(tax, 0) + 1

    # Check CORRECT field names for mappings
    risks_with_broad = sum(1 for r in risks_data if r.get("broad_mappings"))
    risks_with_related = sum(1 for r in risks_data if r.get("related_mappings"))
    risks_with_exact = sum(1 for r in risks_data if r.get("exact_mappings"))
    risks_with_narrow = sum(1 for r in risks_data if r.get("narrow_mappings"))
    risks_with_close = sum(1 for r in risks_data if r.get("close_mappings"))
    risks_with_actions = sum(1 for r in risks_data if r.get("hasRelatedAction"))

    # ============================================================
    # Compile output
    # ============================================================
    all_risk_fields = set()
    for r in risks_data:
        all_risk_fields.update(r.keys())

    all_action_fields = set()
    for a in actions_data:
        all_action_fields.update(a.keys())

    output = {
        "metadata": {
            "total_risks": len(risks_data),
            "total_actions": len(actions_data),
            "taxonomies": sorted(list(taxonomies_seen)),
            "taxonomy_counts": taxonomy_counts,
            "mapping_stats": {
                "risks_with_broad_mappings": risks_with_broad,
                "risks_with_related_mappings": risks_with_related,
                "risks_with_exact_mappings": risks_with_exact,
                "risks_with_narrow_mappings": risks_with_narrow,
                "risks_with_close_mappings": risks_with_close,
                "risks_with_actions": risks_with_actions,
            },
            "all_risk_fields": sorted(list(all_risk_fields)),
            "all_action_fields": sorted(list(all_action_fields)),
        },
        "risks": risks_data,
        "actions": actions_data,
    }

    # ============================================================
    # Save
    # ============================================================
    output_file = "atlas_nexus_full_export.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"Total risks: {len(risks_data)}")
    print(f"Total actions: {len(actions_data)}")
    print(f"Taxonomies: {sorted(list(taxonomies_seen))}")
    print(f"\nRisks per taxonomy:")
    for tax, count in sorted(taxonomy_counts.items()):
        print(f"  {tax}: {count}")
    print(f"\nMapping coverage:")
    print(f"  broad_mappings:   {risks_with_broad}/{len(risks_data)}")
    print(f"  related_mappings: {risks_with_related}/{len(risks_data)}")
    print(f"  exact_mappings:   {risks_with_exact}/{len(risks_data)}")
    print(f"  narrow_mappings:  {risks_with_narrow}/{len(risks_data)}")
    print(f"  close_mappings:   {risks_with_close}/{len(risks_data)}")
    print(f"  hasActions:       {risks_with_actions}/{len(risks_data)}")
    print(f"\nAll risk fields: {sorted(list(all_risk_fields))}")
    print(f"\nUpload {output_file} to continue.")


if __name__ == "__main__":
    main()
