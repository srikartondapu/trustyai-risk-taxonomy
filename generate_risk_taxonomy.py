"""
Generate risk_taxonomy.yaml from Atlas Nexus export.

Reads: atlas_nexus_full_export.json (produced by extract_atlas_data.py)
Writes: taxonomy/data/risk_taxonomy.yaml

Run from the trustyai-risk-taxonomy/ directory:
    python generate_risk_taxonomy.py
"""

import json
import yaml


# Fields to extract from each risk (in display order)
RISK_FIELDS = [
    "id",
    "name",
    "description",
    "isDefinedByTaxonomy",
    "isPartOf",
    "risk_type",
    "tag",
    "descriptor",
    "broad_mappings",
    "related_mappings",
    "exact_mappings",
    "narrow_mappings",
    "close_mappings",
    "hasRelatedAction",
]


def clean_value(value):
    """
    Clean a value for YAML output.
    - None -> omit or empty string
    - Empty list -> []
    - Strings stay as strings
    - Lists stay as lists
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, list):
        # Filter out None items, keep empty list as []
        return [item for item in value if item is not None]
    return value


def extract_risk(raw_risk):
    """Extract the 14 fields we need from a raw risk dict."""
    result = {}
    for field in RISK_FIELDS:
        value = clean_value(raw_risk.get(field))
        result[field] = value
    return result


def main():
    # Load export
    export_file = "atlas_nexus_full_export.json"
    print(f"Reading {export_file}...")
    with open(export_file) as f:
        data = json.load(f)

    raw_risks = data["risks"]
    print(f"Loaded {len(raw_risks)} risks")

    # Extract and clean
    risks = [extract_risk(r) for r in raw_risks]

    # Group by taxonomy for readability
    taxonomy_order = [
        "ibm-risk-atlas",
        "nist-ai-rmf",
        "owasp-llm-2.0",
        "ibm-granite-guardian",
        "ailuminate-v1.0",
        "credo-ucf",
        "mit-ai-risk-repository",
        "mit-ai-risk-repository-causal",
        "ai-risk-taxonomy",
        "shieldgemma-taxonomy",
    ]

    # Sort risks: by taxonomy order, then alphabetically by id within each taxonomy
    def sort_key(risk):
        tax = risk.get("isDefinedByTaxonomy", "")
        try:
            tax_idx = taxonomy_order.index(tax)
        except ValueError:
            tax_idx = 999
        return (tax_idx, risk.get("id", ""))

    risks.sort(key=sort_key)

    # Count per taxonomy
    tax_counts = {}
    for r in risks:
        tax = r.get("isDefinedByTaxonomy", "unknown")
        tax_counts[tax] = tax_counts.get(tax, 0) + 1

    # Build output
    output = {
        "version": "0.1.0",
        "last_updated": "2026-02-09",
        "source": "AI Atlas Nexus (ai-atlas-nexus pip package)",
        "description": (
            "TrustyAI Risk Taxonomy - complete risk catalog. "
            "Contains all risks imported from Atlas Nexus across 10 taxonomies. "
            "This file is the reference layer. Operational mappings "
            "(Garak probes, detectors, evaluations) are in separate files."
        ),
        "total_risks": len(risks),
        "taxonomy_counts": tax_counts,
        "risks": risks,
    }

    # Write YAML
    output_file = "taxonomy/data/risk_taxonomy.yaml"

    # Custom representer to handle None values cleanly
    def represent_none(dumper, _):
        return dumper.represent_scalar("tag:yaml.org,2002:null", "")

    yaml.add_representer(type(None), represent_none)

    # Use default_flow_style=False for readable output
    # But use flow style for short lists to keep file manageable
    class TaxonomyDumper(yaml.Dumper):
        pass

    def represent_list(dumper, data):
        """Use flow style for short lists (fewer than 4 items), block for longer."""
        if len(data) <= 3 and all(isinstance(item, str) and len(item) < 60 for item in data):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

    TaxonomyDumper.add_representer(list, represent_list)
    TaxonomyDumper.add_representer(type(None), represent_none)

    with open(output_file, "w") as f:
        yaml.dump(
            output,
            f,
            Dumper=TaxonomyDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )

    print(f"\nGenerated: {output_file}")
    print(f"Total risks: {len(risks)}")
    print(f"\nRisks per taxonomy:")
    for tax in taxonomy_order:
        count = tax_counts.get(tax, 0)
        print(f"  {tax}: {count}")

    # Quick sanity checks
    ids = [r["id"] for r in risks]
    assert len(ids) == len(set(ids)), "DUPLICATE IDS FOUND!"
    print(f"\nNo duplicate IDs - OK")
    print(f"File written successfully")


if __name__ == "__main__":
    main()
