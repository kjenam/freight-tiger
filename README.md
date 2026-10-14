# Freight Cost Watcher

A monitoring and anomaly detection engine that tracks weekly freight rates (INR per tonne-km) across logistics corridors, flags abnormal cost surges against historical and peer baselines, and attributes verified business reasons from operational context notes.

**Core Architecture Guarantee:** Every metric, threshold check, anomaly flag, and note attribution is computed deterministically in code. The LLM is used strictly as a constrained copywriter to generate concise human explanations, validated against strict grounding checks with zero-dependency template fallbacks.

---

## Getting Started & Execution

### Prerequisites

- **Python 3.10 – 3.12** is recommended. *(Note: C-extensions in dependencies such as `chromadb` and `torch` do not yet have stable wheels for Python 3.14).*
- A free [Groq API Key](https://console.groq.com/keys) if you want LLM-generated reason phrasing (optional; deterministic templates run automatically if no key is provided).

---

### Step-by-Step Setup

#### 1. Clone & Navigate to Repository

```bash
cd freight_tiger
```

#### 2. Create and Activate a Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables (Optional)

Create a `.env` file in the project root to enable Groq LLM-powered explanations:

```bash
# Linux / macOS
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

Open `.env` and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

> **Note:** If `GROQ_API_KEY` is omitted, the entire pipeline still runs end-to-end. Explanations fall back cleanly to deterministic, grammar-checked templates, and all anomaly flags, metrics, and note matches remain identical.

---

### Running the System

#### 1. Run the Main Anomaly Detection Pipeline

```bash
python -m freight_watch.main
```

This runs the complete workflow:
- Loads and validates 2,940 shipment records across 7 corridors.
- Calculates weekly pooled cost per tonne-km, 8-week historical baselines, and peer averages.
- Flags abnormal weeks and attributes valid context notes.
- Generates human-readable explanations via Groq (or template fallback).
- Writes results to:
  - `outputs/flags.csv` — Primary alert table matching the expected output schema.
  - `outputs/run_log.json` — Audit trail containing run metadata, SHA-256 hashes, retrieval mode, and token/cost tracking.

**Optional CLI Flags:**
```bash
python -m freight_watch.main --no-llm         # Force deterministic template explanations
python -m freight_watch.main --no-retrieval   # Run with date-window validation only (skip vector index)
```

#### 2. Run the Unit Test Suite

```bash
pytest
```
Runs test cases covering week bucketing, lookback calculations, peer exclusion, note window matching, and grounding validators.

#### 3. Run Evaluation & Ground Truth Benchmark

```bash
python eval/evaluate.py
```
Scores `outputs/flags.csv` against hand-labeled benchmark cases in `eval/labels.csv` and performs an automated audit over 100% of generated rows.

#### 4. Run Reproducibility Verification

```bash
python scripts/repro_check.py
```
Executes the pipeline 3 times from scratch and verifies byte-for-byte reproducibility across all metrics, flags, and note IDs.

#### 5. Interactive Corridor Q&A (CLI)

Query flagged corridors and operational context using natural language:

```bash
python -m freight_watch.ask "why did Chennai-Bangalore get pricier in March 2025?"
```

Ask broader review questions or inspect evidence:
```bash
python -m freight_watch.ask --show-context "which weeks were unexplained?"
```

---

## How It Works

```
  shipment_records.csv
           │
           ▼
   load.py / metrics.py ───► Weekly Cost/Tonne-Km, 8-Week History %, Peer %
           │
           ▼
      flagging.py       ───► Filter: Rising (>0%) AND Abnormal (Own ≥10% OR Peer ≥20%)
           │
           ▼
  context_notes.csv     ───► Structured Metadata (notes_structured.json)
           │                 + Chroma Vector Store (all-MiniLM-L6-v2)
           │
           ▼
       justify.py       ───► Deterministic Validation (Route + Date Overlap + Cost Direction)
           │                 ├── Matched Note   ──► Flagged: "No (justified)"
           │                 └── No Valid Note  ──► Flagged: "Yes"
           │
           ▼
       explain.py       ───► Guarded Reason Generation (Groq / Template Fallback)
           │
           ▼
   outputs/flags.csv    ───► Final Flagged & Justified Corridor Weeks
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `freight_watch/config.py` | Threshold constants, file paths, embedding and LLM settings. |
| `freight_watch/load.py` | Ingests raw shipments, standardizes column types, and derives Monday-aligned `week_of`. |
| `freight_watch/metrics.py` | Calculates weekly pooled cost per tonne-km, own-history %, and peer-group %. |
| `freight_watch/flagging.py` | Applies candidate anomaly detection rules. |
| `freight_watch/notes.py` | Ingests operational notes and structured attribution metadata. |
| `freight_watch/retrieval.py` | In-memory Chroma vector collection using `all-MiniLM-L6-v2` embeddings for candidate discovery. |
| `freight_watch/justify.py` | Deterministic 3-point rule validator that confirms or rejects candidate notes. |
| `freight_watch/explain.py` | Guarded reason phrasing using Groq `openai/gpt-oss-20b` with automated grounding checks. |
| `freight_watch/main.py` | End-to-end orchestration CLI and run logger. |
| `freight_watch/ask.py` | Interactive CLI for question answering over flagged rows and context notes. |
| `scripts/extract_notes.py` | One-time metadata extraction utility (outputs reviewed and committed to `notes_structured.json`). |
| `scripts/repro_check.py` | 3-run deterministic consistency verification test. |
| `eval/evaluate.py` | Benchmark evaluation and comprehensive output auditing. |

---

## Dataset Summary

The dataset comprises **2,940 shipments** spanning **7 corridors** and **104 consecutive weeks** (2024-01-01 to 2025-12-22). Every corridor has continuous activity across all weeks with no missing data or duplicate entries.

| Corridor Type | Corridors | Characteristics |
|---|---|---|
| **Short** | Delhi-Jaipur, Mumbai-Pune | 2 routes; higher baseline rate variance per km. |
| **Medium** | Ahmedabad-Mumbai, Chennai-Bangalore, Kolkata-Bhubaneswar | 3 routes; balanced peer comparison baseline. |
| **Long** | Delhi-Chennai, Mumbai-Delhi | 2 routes; lower rates per tonne-km due to long-haul economics. |

---

## Mathematical Formulation & Anomaly Rules

### 1. Week Alignment
Weeks run Monday through Sunday. Every shipment is bucketed to its week's Monday:
$$\text{week\_of} = \text{shipment\_date} - \text{weekday}(\text{shipment\_date})$$
*(e.g., a Sunday shipment on 2024-01-07 belongs to `2024-01-01`).*

### 2. Weekly Cost Per Tonne-Km
For each corridor $r$ and week $w$, the unit freight cost is computed as a **pooled ratio** over all shipments in that week:
$$\text{cost}_{r,w} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$
*Note: This is a volume-weighted pooled ratio across the corridor-week, reproducing target sample values (4.17, 3.29, 3.98) exactly.*

### 3. Comparison vs. Own History
Measures the percentage change against the corridor's own mean cost over the **prior 8 weeks** (strictly lookback, no current-week lookahead):
$$\text{own\%} = \frac{\text{cost}_{r,w}}{\text{mean}(\text{cost}_{r, w-8 \dots w-1})} - 1$$
- Weeks 2 through 8 of 2024 utilize available history (1 to 7 weeks) without padding or artificial extrapolation.
- Week 1 (2024-01-01) has zero prior history, making `vs_own_history` undefined; it is never flagged.

### 4. Comparison vs. Similar Routes (Peers)
Measures the percentage change against contemporaneous costs on other corridors of the **same route type** in the identical week:
$$\text{peer\%} = \frac{\text{cost}_{r,w}}{\text{mean}_{r' \ne r,\ \text{type}(r') = \text{type}(r)}(\text{cost}_{r', w})} - 1$$
The corridor under evaluation is strictly excluded from its own peer average.

### 5. Anomaly Flagging Criteria
A corridor-week is surfaced as an alert candidate if and only if:
1. **Cost is rising:** `vs_own_history > 0.0%`
2. **Cost is abnormal:** `vs_own_history >= +10.0%` **OR** `vs_similar_routes >= +20.0%`

**Engineering Rationale:**
- **The 10% Own-History Threshold** isolates sharp corridor-specific shocks (e.g., Chennai-Bangalore cyclone flooding, Ahmedabad-Mumbai festival surcharges, Delhi-Jaipur capacity squeezes).
- **The 20% Peer Threshold** isolates creeping cost inflation where rolling 8-week lookbacks gradually absorb rising costs (e.g., Mumbai-Pune climbing relative to Delhi-Jaipur).
- **Why Peer is 20% (and not lower):** Mumbai-Pune maintains an inherent structural operating cost ~12% higher than Delhi-Jaipur throughout normal weeks. A lower peer threshold would trigger continuous false positives on healthy operations.
- **The "Rising" Gate:** A corridor running cheaper than its own past average does not warrant an alert, even if peer differences fluctuate.

---

## Note Attribution & Grounding Engine

### Deterministic Validation Rules
Context notes are never attributed solely based on vector similarity. A candidate note justifies a cost rise **if and only if all three criteria pass**:

1. **Corridor Match:** Note's `applies_to` matches the corridor exactly, or is marked `All Routes`.
2. **Temporal Overlap:** The note's effective window overlaps with the Monday–Sunday shipment week.
3. **Cost Directionality:** The note confirms an upward cost impact (`raises_cost == True`).

If multiple notes pass:
- Route-specific notes take strict priority over generic `All Routes` notes.
- The note closest in date to the corridor week breaks any remaining ties.

### Structured Note Registry

Operational context notes are pre-processed and committed in `data/notes_structured.json` with explicit validity windows:

| Note ID | Applies To | Effective Window | Raises Cost? | Operational Context |
|---|---|---|---|---|
| **N001** | Chennai-Bangalore | 2025-02-24 to 2025-03-08 | Yes | Severe flooding, NH detours, bridge damage. |
| **N002** | Ahmedabad-Mumbai | 2025-01-20 to 2025-01-26 | Yes | Regional festival capacity surcharge. |
| **N003** | All Routes | 2025-05-05 to 2025-06-02 | Yes | Nationwide diesel price increase (~5-7%). |
| **N004** | All Routes | 2024-03-11 | No | Port congestion on routes outside this dataset. |
| **N005** | Mumbai-Delhi | 2024-07-29 | No | Landslides noted; costs reported unaffected. |
| **N006** | All Routes | 2025-09-22 | No | Demand stable; standard operating conditions. |
| **N007** | Delhi-Jaipur | 2024-05-20 | No | Highway resurfacing improving efficiency. |
| **N008** | Kolkata-Bhubaneswar | 2025-06-09 | No | Seasonal rainfall within normal operational limits. |
| **N009** | Chennai-Bangalore | 2025-03-17 | No | Flood recovery completed; rates normalized. |
| **N010** | All Routes | 2025-10-27 | No | Regulatory compliance costs absorbed by carriers. |

*Window Policy:* Notes lacking explicit expiry receive a default 4-week window (`DEFAULT_NOTE_WINDOW_WEEKS = 4`).

### LLM Guardrails & Fallbacks

- **Zero Generative Verdicts:** The model is never asked whether a row is justified. It only crafts a concise, professional phrasing of the upstream deterministic verdict.
- **Automated Grounding Verification:**
  - Justified rows must reference the matched note ID and cannot reference any other note.
  - Unexplained rows are prevented from asserting justification or attributing causality.
- **Fail-Safe Fallback:** If API limits, network timeouts, or grounding checks fail, phrasing falls back immediately to verified deterministic templates matching the exact sample style.

---

## Evaluation & Benchmark Results

### 1. Ground Truth Benchmark (`eval/evaluate.py`)

Scored against 20 hand-labeled route-weeks across 7 corridors:

| Evaluation Metric | Score | Detail |
|---|---|---|
| **Flag Precision** | **1.00** (11/11) | 100% of surfaced candidate rows warranted an alert. |
| **Flag Recall** | **1.00** (11/11) | Zero missed anomalies across all abnormal weeks. |
| **Review Precision** | **1.00** (7/7) | All rows flagged as "Yes" (unexplained) are truly unresolved. |
| **Review Recall** | **1.00** (7/7) | All unresolved rows were correctly routed for human review. |
| **Verdict Accuracy** | **1.00** (20/20) | 100% exact 3-way match (`Yes`, `No (justified)`, `Not flagged`). |
| **Note-Match Accuracy** | **1.00** (4/4) | Perfect attribution for all justified corridor-weeks. |
| **Full Output Audit** | **PASS** | 20/20 output rows verified for math accuracy and note validity. |

### 2. Reproducibility Check (`scripts/repro_check.py`)

Executing 3 independent pipeline runs from clean state confirms:
- **Numerical & Flag Columns:** 100% byte-identical across all runs (`route`, `week_of`, `cost_per_tonne_km`, `vs_own_history`, `vs_similar_routes`, `flagged`, `matched_note_id`).
- **Explanation Stability:** Consistent semantic output with 0 verdict discrepancies.

### 3. Pipeline Performance & Cost Log (`outputs/run_log.json`)

| Run Metric | Production Value |
|---|---|
| **Total Shipments Processed** | 2,940 records |
| **Corridor-Weeks Evaluated** | 728 weeks |
| **Total Rows Surfaced** | 20 rows (16 Unexplained `Yes`, 4 Justified `No`) |
| **LLM Model** | `openai/gpt-oss-20b` via Groq |
| **LLM Invocations** | 20 calls (strictly 1 call per output row; 0 on raw shipments) |
| **Token Usage** | 9,461 prompt tokens, 1,493 completion tokens (10,954 total) |
| **Total Run Cost** | **$0.00116 USD** (free tier on Groq) |
| **Template Fallback Rate** | 0% (all 20 LLM outputs passed grounding checks) |

---

## Project Structure

```
freight_tiger/
├── data/
│   ├── shipment_records.csv         # 2,940 shipment transactions
│   ├── context_notes.csv            # 10 unstructured operational notes
│   ├── notes_structured.json        # Verified note metadata registry
│   └── sample_output_format_v2.csv  # Target schema & formatting reference
├── eval/
│   ├── evaluate.py                  # Evaluation & audit runner
│   └── labels.csv                   # 20 ground-truth labeled route-weeks
├── freight_watch/
│   ├── config.py                    # Global constants & thresholds
│   ├── load.py                      # Data validation & Monday week-bucketing
│   ├── metrics.py                   # Pooled cost/tonne-km & relative baselines
│   ├── flagging.py                  # Anomaly detection filtering rules
│   ├── notes.py                     # Note models & loader
│   ├── retrieval.py                 # In-memory Chroma vector index
│   ├── justify.py                   # Deterministic attribution validator
│   ├── explain.py                   # Groq LLM client, prompt & guardrails
│   ├── ask.py                       # CLI assistant for natural language queries
│   └── main.py                      # Pipeline entrypoint
├── outputs/
│   ├── flags.csv                    # Final generated alert output
│   ├── run_log.json                 # Execution audit log & token accounting
│   ├── eval_report.txt              # Latest benchmark test results
│   └── repro_report.txt             # Latest 3-run reproducibility report
├── scripts/
│   ├── extract_notes.py             # One-time metadata extraction helper
│   └── repro_check.py               # 3-run byte-equality verification test
├── tests/                           # Pytest unit tests
├── requirements.txt                 # Pinned dependencies
└── README.md
```

---

## Key Design Decisions & Trade-offs

1. **Deterministic Logic vs. Generative Verdicts:**
   Allowing an LLM to decide whether a surge is "justified" introduces non-determinism, drift, and hallucination risks. Here, 100% of the anomaly math and note attribution rules are executed in deterministic Python. The LLM's role is strictly confined to clear phrasing.

2. **Fixed Thresholds vs. Statistical (Z-Score) Models:**
   A rolling z-score model can struggle when baseline volatility changes across seasons or when route samples are small. The combination of a +10% own-history gate and a +20% peer-corridor gate aligns directly with how freight operators evaluate rate escalations and avoids false triggers on regular corridor variance.

3. **Hybrid Note Matching (Semantic Retrieval + Deterministic Filtering):**
   While a pure date/route filter works well for small note sets, vector embeddings (`all-MiniLM-L6-v2`) enable the system to scale to hundreds of unstructured field notes. Crucially, the vector index only serves as a candidate retriever—the deterministic validator always retains veto power over the final attribution.

4. **Bounded Lookback (8 Weeks) vs. Lifetime Baselines:**
   An 8-week lookback adapts dynamically to gradual macro shifts (such as fuel price trends or seasonal shifts) without letting acute multi-week disruptions permanently skew historical baselines.
