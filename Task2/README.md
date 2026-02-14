# Task #2: Merging PatentsView, DISCERN, and Clinical Trials

**Author:** Edward Jung
**Date:** February 14, 2026

---

## Objective

Construct a **firm-year dataset** of AI-related patent applications for biopharma firms conducting clinical trials.

---

## Primary Deliverable

**File:** `firm_year_patents.csv`

**Structure:** One row per firm-year (gvkey × year)

**Columns:**
- `gvkey` - Firm identifier (Compustat)
- `year` - Application year (2000-2025)
- `total_applications` - Total patent applications filed
- `ai_applications` - AI-related applications
- `ai_share` - Proportion of AI patents (ai / total)
- `ai_dummy` - Binary indicator (1 if ≥1 AI patent)

---

## Methodology

### Data Sources
1. **PatentsView** - Patent application data (2000-2025)
2. **DISCERN 2** - Patent assignee to GVKEY mapping
3. **Clinical Trials** - Provided sample with NCT IDs, sponsors, GVKEY

### AI Classification
**Dual approach** for identifying AI-related patents:

**Method 1: CPC Codes (High Precision)**
- G06N family: G06N3 (neural networks), G06N5 (knowledge-based), G06N7 (probabilistic), G06N10 (quantum), G06N20 (machine learning)
- Examiner-assigned, standardized
- Only available for granted patents

**Method 2: Keywords (High Recall)**
- 40+ AI-related terms in titles/abstracts
- Examples: machine learning, deep learning, neural network, reinforcement learning
- Works for both granted and pending applications

**Combined:** Patent flagged as AI if **either** method detects it

### Processing Workflow
1. Load clinical trials → extract sponsor-GVKEY mappings
2. Import PatentsView applications (2000-2025)
3. Map applicants to GVKEY via name standardization
4. Classify AI patents using CPC codes + keywords
5. Aggregate to firm-year level
6. Export clean summary table

---

## Repository Contents

### Analysis Notebook
**`task2_patents_discern_merge.ipynb`**
- Complete reproducible workflow
- Memory-efficient processing using DuckDB
- Documented methodology
- Validation checks throughout

### Documentation
**`DOCUMENTATION.md`**
- Technical specifications
- Data architecture design
- AI classification methodology
- Memory efficiency strategies
- Validation approach

### Input Data
**`clinical_trial_sample (1).csv`**
- Provided clinical trials dataset
- Contains: NCT IDs, sponsor names, GVKEY, start years, phases

### Task Description
**`RA Task #2.pdf`**
- Original task requirements

---

## Output Files

After running the notebook:

**Primary:**
- `firm_year_patents.csv` - Aggregated firm-year dataset (main deliverable)

**Alternative:**
- `firm_year_merged.csv` - Same as above + clinical trial metrics (num_trials, avg_phase)

**Optional:**
- `patent_level_dataset.csv` - Detailed patent classifications (for validation)
- `task2_patents.ddb` - DuckDB database (intermediate)

---

## Key Features

- **Focus on applications** (not just grants) for temporal alignment with trials
- **Time period:** 2000-2025 application years
- **Memory-efficient:** Handles 10-15 GB PatentsView data on standard hardware
- **Dual AI classification:** Combines precision (CPC) with recall (keywords)
- **Clean panel structure:** One observation per firm-year

---

## Technical Requirements

**Required:**
- Python 3.9+
- pandas, numpy, duckdb

**Optional:**
- rapidfuzz (fuzzy matching)
- psutil (memory monitoring)

**Runtime:** ~30-60 minutes (includes data download)

---

## Usage

```bash
jupyter notebook task2_patents_discern_merge.ipynb
```

Run all cells sequentially. PatentsView tables download automatically.

---

## Data Architecture

**Two-layer design:**

1. **Patent-level** (internal) - Used for AI classification and validation
2. **Firm-year** (deliverable) - Aggregated summary for analysis

This approach enables transparent classification while delivering analysis-ready data.

---

## Validation

- Match rate: >50% of clinical trial firms matched to patents
- AI share: 1-10% typical for 2000-2025 period
- Temporal trends: AI patents increase over time
- Sample validation: Manual review of classifications

---

**For detailed technical documentation, see `DOCUMENTATION.md`**
