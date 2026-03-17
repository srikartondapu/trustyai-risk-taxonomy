# 🛡️ TrustyAI Risk Taxonomy

A structured risk taxonomy that maps AI safety risks to operational actions -- which vulnerability probes to run, which detectors to deploy, which benchmarks to evaluate, and where the coverage gaps are.

It is model-agnostic and application-agnostic. Whether you're securing an LLM chatbot, a RAG pipeline, or an agentic system, the taxonomy provides risk assessments, custom mitigation recommendations based on your domain and use case, and compliance reporting against whichever standards you need to adhere to (OWASP LLM Top 10, NIST AI RMF, and others).

## 🔍 The Problem

Vulnerability scanners, runtime guardrails, evaluation frameworks, and guardrail optimizers all operate independently. When a Garak scan flags `dan.DanInTheWild` at 57% pass rate, there is no structured way to:

- Know that maps to a **jailbreak** risk (not toxicity, not PII)
- Find that we have a **jailbreak detector** available and what threshold it uses
- Determine which **evaluation benchmarks** measure jailbreak severity
- Trace it back to **OWASP LLM01** and **NIST AI RMF** requirements
- Check if the detector actually **reduces the attack success rate** after deployment

That mapping lives as tribal knowledge. This taxonomy makes it queryable.

## 🏗️ How It Works

The taxonomy builds on IBM's [Atlas Nexus](https://github.com/IBM/risk-atlas-nexus), which provides 546 risk definitions across 10 industry taxonomies (IBM Risk Atlas, NIST AI RMF, OWASP LLM Top 10, Granite Guardian, ML Commons, and others) with cross-taxonomy mappings between them.

On top of that reference data, this project adds an **operational layer**: probe-to-risk mappings, detector configurations with thresholds, evaluation benchmarks, tiered coverage tracking, and domain-specific profiles. Atlas Nexus tells you what the risks are. This taxonomy tells you what to do about them.

- **Domain profiles** adjust thresholds and priorities per use case -- a healthcare patient chatbot gets stricter hallucination controls than a general-purpose assistant
- **Compliance traceability** maps any of the 10 supported standards down to operational dimensions, showing which requirements are covered, at what level, and where the gaps are

```
Vulnerability Scan -> Taxonomy Lookup -> Mitigation Recommendation -> Deploy Guardrail
     -> Re-scan -> Effectiveness Write-back -> Optimizer
```

## 🚀 Quick Start

```bash
python3 -m venv riskenv
source riskenv/bin/activate
pip install .

# Run tests
python test_taxonomy.py      # 32 tests
python test_export.py        # 57 tests (some skip without scan data, 0 fail)

# Run the governance loop (CLI)
python demo_governance_loop.py

# Run the Streamlit demo
pip install ".[demo]"
streamlit run demo_app.py --server.headless true
# Open http://localhost:8501
```

Everything works out of the box without scan data. The CLI governance loop uses built-in mock scan results and labels them as `MOCK DATA`. The Streamlit demo loads all taxonomy and coverage tabs; tabs that depend on scan files show a warning and degrade gracefully.

To use real Garak scan data, point the CLI at a report file:

```bash
python demo_governance_loop.py --report path/to/scan.report.jsonl
```

## 📁 Repository Structure

```
taxonomy/                       # Core library
  query.py                      # TaxonomyManager -- single query interface (19 public methods)
  __init__.py
  data/
    risk_taxonomy.yaml          # 546 risks from 10 taxonomies (Atlas Nexus extract)
    optimizer_dimensions.yaml   # 5 operational dimensions with primary/related risks
    risk_to_garak.yaml          # 24 probe mappings with AVID/OWASP tags
    risk_to_mitigations.yaml    # Two-hop: risk -> mitigation type -> detector options
    risk_to_eval.yaml           # Evaluation benchmarks per risk
    domain_profiles.yaml        # 7 domain/use-case profiles (override-only pattern)

test_taxonomy.py                # 32 tests -- risk queries, reverse lookups, coverage, compliance
test_export.py                  # 57 tests -- CSV generation, effectiveness computation

demo_app.py                     # Streamlit demo (10 tabs, full governance loop)
demo_governance_loop.py         # CLI version of the governance loop

parse_garak.py                  # Parse Garak .report.jsonl files into structured results
export.py                       # Generate optimizer CSVs from taxonomy data + scan results
compare_scans.py                # Compare before/after guardrail scans, compute effectiveness
extract_atlas_data.py           # Extract risk data from Atlas Nexus Python library
generate_risk_taxonomy.py       # Convert Atlas Nexus export to risk_taxonomy.yaml
```

`extract_atlas_data.py` and `generate_risk_taxonomy.py` are data regeneration scripts. They require `ai-atlas-nexus` (`pip install ai-atlas-nexus`) and are not needed for normal use -- the extracted data is already in `taxonomy/data/`.

## ⚙️ Architecture

```
Atlas Nexus (546 risks, 10 taxonomies, cross-taxonomy SSSOM mappings)
  |
  | extract                          REFERENCE LAYER -- risk definitions
  v
risk_taxonomy.yaml
  |
  +-- optimizer_dimensions.yaml
  +-- risk_to_garak.yaml             OPERATIONAL LAYER -- our data
  +-- risk_to_mitigations.yaml
  +-- risk_to_eval.yaml
  +-- domain_profiles.yaml
  |
  |-- TaxonomyManager (query.py -- in-memory indexes, <10ms queries)
  |
  v
Consumers: scanners, orchestrators, eval frameworks, optimizer
```

### Key Design Decisions

- **Two-hop mitigation**: risk -> mitigation type -> detector options. Supports multiple detectors per risk with different cost/latency/accuracy tradeoffs. Adding a detector is a YAML edit.
- **Tiered coverage**: full (active detector + measured effectiveness), baseline (detector available, not yet measured), gap (no detector). Not binary.
- **Override-only profiles**: domain profiles only specify overrides from base defaults. Healthcare overrides hallucination to CRITICAL; everything else inherits.
- **Garak tags as validation evidence**: we own the probe-to-risk mappings. Garak's AVID/OWASP tags are stored alongside as corroboration, not as the mapping source.

## 📊 Current State

| Dimension | Coverage | Measured Effectiveness |
|-----------|----------|----------------------|
| Harmful Content | ✅ Full | 100% |
| Jailbreak | ✅ Full | 89% |
| PII Leakage | ✅ Full | 100% |
| Hallucination | 🔶 Baseline | Not yet measured |
| Bias & Fairness | ⬜ Gap | No detector available |

5 operational dimensions, 11 detectors (4 active + 7 built-in), 24 Garak probes mapped, 7 domain profiles, 89 tests passing.

## 🗺️ What's Next

- Expand from 5 to ~10 operational dimensions (model extraction, copyright, overreliance, agentic safety)
- Tag all 546 risks with actionability level (runtime / training-time / governance)
- Model causal chains between dimensions (jailbreak -> content_safety, jailbreak -> pii)
- Switch from static Atlas Nexus extract to runtime dependency
- Add JSON Schema validation in CI
