# Task #2: Patents, DISCERN, and Clinical Trials Integration
## Technical Documentation

**Author:** Edward Jung
**Date:** February 14, 2026
**Objective:** Construct a firm-year dataset of AI-related patent applications for biopharma firms conducting clinical trials

---

## Table of Contents

1. [Overview](#overview)
2. [Data Architecture](#data-architecture)
3. [AI Classification Methodology](#ai-classification-methodology)
4. [Implementation Details](#implementation-details)
5. [Memory Efficiency Strategy](#memory-efficiency-strategy)
6. [Output Specifications](#output-specifications)
7. [Validation Approach](#validation-approach)

---

## Overview

### Objective
Build a firm-year aggregated dataset (gvkey × year) containing patent application metrics for firms conducting clinical trials, with specific focus on AI-related innovations.

### Key Design Decisions

**Focus on Applications, Not Grants**
- Patent applications capture earliest innovation timing
- Applications align temporally with clinical trial initiation
- Grants have 1-3 year lag and exclude pending applications

**Time Period**
- Application years: 2000-2025
- Aligns with clinical trial start years in provided dataset

**Dual AI Classification**
- Method 1: CPC codes (high precision)
- Method 2: Keyword filtering (high recall)
- Combined approach maximizes coverage

---

## Data Architecture

### Two-Layer Design

#### Layer 1: Patent-Level (Internal Processing)
Intermediate working dataset used for AI classification.

**Purpose:**
- Enable AI classification via CPC codes and keywords
- Support validation and quality checks
- Maintain audit trail of classification decisions

**Structure:**
- One row per patent application
- Contains: application_id, patent_id, filing_date, gvkey, is_ai, ai_method, title, abstract
- Used internally; can be exported for validation

#### Layer 2: Firm-Year (Final Deliverable)
Aggregated summary table for analysis.

**Purpose:**
- Research-ready dataset for regression analysis
- Clean panel data structure
- One observation per firm-year

**Structure:**

| Column | Type | Description |
|--------|------|-------------|
| gvkey | string | Firm identifier (Compustat) |
| year | integer | Calendar year (2000-2025) |
| total_applications | integer | Total patent applications filed |
| ai_applications | integer | AI-related applications |
| ai_share | float | ai_applications / total_applications |
| ai_dummy | integer | 1 if ai_applications ≥ 1, else 0 |

**Rationale for Two Layers:**
- Patent-level: Transparency and validation
- Firm-year: Analysis efficiency and standard panel format
- Reproducible aggregation from detailed to summary level

---

## AI Classification Methodology

### Dual Classification Approach

AI patents are identified using **two complementary methods**, with a patent flagged as AI-related if **either** method detects it.

### Method 1: CPC Classification Codes (High Precision)

**Approach:** Identify patents with AI-related Cooperative Patent Classification codes

**AI-Related CPC Codes:**
```
G06N - Computing based on specific computational models
├── G06N3  - Neural networks
├── G06N5  - Knowledge-based models
├── G06N7  - Probabilistic/fuzzy logic
├── G06N10 - Quantum computing
└── G06N20 - Machine learning
```

**Advantages:**
- Examiner-assigned (authoritative)
- Internationally standardized
- High precision (~90-95%)
- No false positives from unrelated terminology

**Limitations:**
- Only available for granted patents
- Pending applications lack CPC codes
- Classification lag (examiners assign codes during review)

### Method 2: Keyword-Based Filtering (High Recall)

**Approach:** Search patent titles and abstracts for AI-related terminology

**Keyword Categories (40+ terms):**

*Core ML Terms:*
- machine learning, deep learning, neural network, artificial intelligence

*Learning Paradigms:*
- supervised learning, unsupervised learning, reinforcement learning, transfer learning

*Specific Models:*
- random forest, gradient boosting, support vector machine, bayesian network, LSTM, transformer

*Applications:*
- computer vision, natural language processing, image recognition, predictive model

*Techniques:*
- feature extraction, dimensionality reduction, classification algorithm

**Advantages:**
- Works for both granted and pending applications
- Captures emerging AI terminology
- Higher recall than CPC alone
- Flexible and updatable

**Limitations:**
- May include false positives (~20-30%)
- Requires careful keyword curation
- Context-dependent matches

### Combined Classification Logic

```python
is_ai = (is_ai_cpc OR is_ai_keyword)

ai_method = {
    'cpc': CPC only,
    'keyword': Keyword only,
    'both': Both methods detected AI
}
```

**Interpretation:**
- **CPC only:** High confidence (granted patents)
- **Keyword only:** Moderate confidence (may include pending)
- **Both:** Very high confidence
- **ai_method** field enables sensitivity analysis

---

## Implementation Details

### Data Sources

**1. PatentsView Bulk Downloads**
- Base URL: https://s3.amazonaws.com/data.patentsview.org/download/
- Tables used:
  - `g_application` (~2-3 GB) - Core application data
  - `pg_applicant_not_disambiguated` (~1-2 GB) - Applicant names
  - `g_cpc_current` (~4 GB) - CPC classification codes
  - `g_patent_abstract` (~6 GB) - Patent abstracts for keyword search

**2. DISCERN 2**
- URL: https://zenodo.org/records/13619821
- Purpose: Map patent assignees to Compustat GVKEY
- Handles name variations and time-varying firm identifiers

**3. Clinical Trials Dataset**
- Provided: clinical_trial_sample.csv
- Contains: NCT IDs, sponsor names, GVKEY, start years, phases

### Processing Workflow

```
1. Load Clinical Trials
   - Extract unique sponsor names and GVKEYs
   ↓
2. Import PatentsView Applications (2000-2025)
   - Filter by filing date
   - ~10-15 GB raw data
   ↓
3. Map Applicants to GVKEY
   - Name standardization (clean legal suffixes, punctuation)
   - Match to clinical trial sponsors
   - (Optional: DISCERN 2 integration for broader coverage)
   ↓
4. Classify AI Patents
   - Load CPC codes → identify G06N* codes
   - Load titles/abstracts → keyword search
   - Mark is_ai if either method flags
   ↓
5. Aggregate to Firm-Year
   - Group by gvkey × year
   - Count total_applications, ai_applications
   - Calculate ai_share, ai_dummy
   ↓
6. Export Deliverables
   - firm_year_patents.csv (primary)
   - firm_year_merged.csv (with clinical trials)
```

### Name Standardization Function

For mapping patent applicants to GVKEY:

```python
def clean_org_name(name):
    """Standardize organization names for matching."""
    name = name.lower()

    # Remove legal suffixes
    suffixes = ['inc', 'corp', 'ltd', 'llc', 'plc', 'sa', 'ag', 'gmbh']
    for suffix in suffixes:
        name = re.sub(rf'\b{suffix}\.?\b', '', name)

    # Remove punctuation and extra whitespace
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    name = ' '.join(name.split())

    return name.strip()
```

---

## Memory Efficiency Strategy

Given large PatentsView files (10-15 GB total), several strategies ensure processing on standard hardware:

### 1. DuckDB Pre-Filtering
```python
# SQL filtering before pandas import
con.execute("""
    SELECT * FROM g_application
    WHERE filing_date BETWEEN '2000-01-01' AND '2025-12-31'
""").df()
```
**Benefit:** Only loads relevant years into memory

### 2. Optimized Data Types
```python
dtypes = {
    'gvkey': 'category',      # 50-70% memory reduction
    'year': 'int16',          # vs int64
    'is_ai': 'bool'           # vs object
}
```

### 3. Column Pruning
Load only necessary columns instead of entire tables

### 4. Chunked Processing
For extremely large datasets:
```python
for chunk in pd.read_csv('file.tsv', chunksize=100000):
    process(chunk)
```

### 5. Year-by-Year Processing
Process one year at a time if memory constrained:
```python
for year in range(2000, 2026):
    year_data = load_year(year)
    process(year_data)
    del year_data  # Free memory
```

**Expected Memory Usage:** 2-4 GB (vs 15-20 GB naive approach)

---

## Output Specifications

### Primary Deliverable: firm_year_patents.csv

**Format:** CSV (comma-separated values)
**Structure:** One row per gvkey-year combination
**Size:** ~500-2000 rows (depending on firms matched)

**Example:**
```csv
gvkey,year,total_applications,ai_applications,ai_share,ai_dummy
8530,2015,342,12,0.035,1
8530,2016,389,18,0.046,1
9775,2015,298,5,0.017,1
9775,2016,312,8,0.026,1
```

**Column Definitions:**

- **gvkey:** Compustat firm identifier (string)
- **year:** Application filing year, 2000-2025 (integer)
- **total_applications:** Count of all patent applications filed by firm in year (integer)
- **ai_applications:** Count of AI-related applications (integer)
- **ai_share:** Proportion of applications that are AI-related (float, 0-1)
- **ai_dummy:** Binary indicator for any AI activity (1 = firm has ≥1 AI patent, 0 = none)

### Alternative: firm_year_merged.csv

Same as primary deliverable plus clinical trial metrics:
- **num_trials:** Count of clinical trials started by firm in year
- **avg_phase:** Average trial phase (1-4)

**Use Case:** If analysis requires simultaneous examination of patent and trial activity

### Supporting Files

**Reproducible Methodology:**
- `task2_patents_discern_merge.ipynb` - Complete workflow with documentation

**Optional (for validation):**
- `patent_level_dataset.csv` - Detailed patent-level classifications

---

## Validation Approach

### Data Quality Checks

**Match Rate:**
- Target: >50% of clinical trial firms matched to patent data
- Calculated: (firms with patents) / (total clinical trial firms)
- Low match rate suggests need for DISCERN 2 integration

**AI Share Distribution:**
- Expected: 1-10% of applications flagged as AI (typical for 2000-2025)
- Outliers investigated manually

**Temporal Trends:**
- AI patents should increase over time (2000-2025)
- Sudden spikes/drops investigated

### Classification Validation

**Sample-Based Review:**
1. Randomly sample 50-100 patents from each method
2. Manually review titles/abstracts
3. Calculate precision = true positives / total sampled
4. Refine keywords based on false positives

**Cross-Validation:**
- Compare to known AI patent portfolios (IBM, Google)
- Check against external datasets (Google Patents, Lens.org)
- Verify temporal trends match industry reports

### Aggregation Checks

**Firm-Year Consistency:**
- Sum of firm-year totals = count of unique patents
- No duplicate gvkey-year combinations
- ai_share = ai_applications / total_applications (verified)
- ai_dummy = 1 iff ai_applications > 0

**Merge Quality (if using firm_year_merged.csv):**
- Firms from both datasets preserved (outer join)
- No spurious matches
- Missing values handled appropriately (0 for counts)

---

## Technical Environment

**Required:**
- Python 3.9+
- pandas ≥1.5.0
- numpy ≥1.23.0
- duckdb ≥0.9.0

**Optional:**
- rapidfuzz ≥2.0.0 (fuzzy matching for DISCERN 2)
- psutil ≥5.9.0 (memory monitoring)

**Runtime Estimates (8GB RAM, SSD):**
- Data download: 10-30 minutes (one-time)
- Data import: 5-10 minutes
- AI classification: 5-15 minutes
- Aggregation: 1-2 minutes
- **Total:** ~30-60 minutes

---

## References

**Data Sources:**
- PatentsView: https://patentsview.org/download/data-download-tables
- DISCERN 2: https://zenodo.org/records/13619821
- CPC Codes: https://www.cooperativepatentclassification.org/

**Methodology:**
- Fujii & Managi (2018): "Trends and priority shifts in artificial intelligence technology invention"
- OECD AI Patents Report (2021): https://www.oecd.org/sti/measuring-innovation-in-artificial-intelligence.pdf

---

**Last Updated:** February 14, 2026
