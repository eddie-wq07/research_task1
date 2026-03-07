# Clinical Trial Firms with AI Patents Analysis

**Objective:** Compute the percentage of companies involved in clinical trials that hold at least one AI-related patent.

## Overview

This analysis computes the percentage of clinical trial firms that hold at least one AI-related patent, defined as patents with CPC (Cooperative Patent Classification) codes starting with "G06N".

## AI Patent Definition

**AI patents** are identified by CPC codes starting with **G06N**, which includes:
- **G06N3**: Neural networks
- **G06N5**: Knowledge-based models
- **G06N7**: Probabilistic/fuzzy logic
- **G06N10**: Quantum computing
- **G06N20**: Machine learning

## Data Pipeline

The analysis follows this data pipeline:

1. **Clinical Trials** → Obtain GVKEY for firms involved in clinical trials
2. **DISCERN Firm Panel** → Map GVKEY to `permno_adj` (with fiscal year filtering: 2000-2021)
3. **DISCERN Patent Grants** → Map `permno_adj` to `patent_id`
4. **PatentsView API** → Retrieve CPC classifications for each patent
5. **Filter** → Identify patents with CPC codes starting with "G06N"
6. **Aggregate** → Flag firms with at least one AI patent
7. **Compute** → Calculate percentage

## Time Window

- **Fiscal Year Range**: 2000-2021
- **Rationale**: DISCERN data ends in 2021; fiscal years 2022-2023 are excluded

## Data Sources

### Input Data
1. **Clinical Trials**: `../Task2 Outdated/clinical_trial_sample (1).csv`
   - Contains: NCT IDs, sponsor names, GVKEYs, start dates, phases

2. **DISCERN 2.0.1** (from `../Task2 Outdated/Task2 Outdated V2/DISCERN 2_0_1/output_files/stata_files/`):
   - `discern_firm_panel_1980_2021.dta`: Maps GVKEY to permno_adj by fiscal year
   - `discern_pat_grant_1980_2021.dta`: Maps permno_adj to patent_id

3. **PatentsView Bulk Downloads**: https://patentsview.org/download/data-download-tables
   - Provides CPC classifications for patent IDs
   - Note: The PatentsView Legacy API was deprecated on May 1, 2025
   - We now use bulk data downloads instead of API queries

### Output Files
The notebook generates:
1. **clinical_trial_firms_ai_patents.csv**: Firm-level results with AI patent counts
2. **summary_statistics.csv**: Aggregate statistics and percentages
3. **cpc_codes_checkpoint.csv**: CPC data cache (for re-running without API queries)

## Key Output Metrics

The analysis computes three key metrics:

1. **Total number of clinical trial firms**: Count of unique GVKEYs in clinical trial dataset
2. **Number of firms with AI patents**: Count of firms with ≥1 patent having G06N CPC code
3. **Percentage of firms with AI patents**: (Firms with AI patents / Total firms) × 100

## Usage Instructions

### Prerequisites

Install required Python packages:

```bash
pip install pandas numpy requests tqdm
```

### Running the Analysis

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook clinical_trials_ai_patents_analysis.ipynb
   ```

2. **Run all cells sequentially** (Cell → Run All)

3. The notebook will:
   - Load clinical trial and DISCERN data
   - Query PatentsView API for CPC codes (may take 30-60 minutes)
   - Filter for G06N patents
   - Compute and display final statistics

### Important Notes

- **Bulk Data Download**: The notebook downloads CPC classification data from PatentsView bulk downloads (~4GB file). This may take 10-30 minutes on first run, but subsequent runs use cached data.

- **API Deprecation**: The PatentsView Legacy API was shut down on May 1, 2025. This notebook uses bulk data downloads instead.

- **Checkpoint Files**: CPC data is saved to `cpc_codes_checkpoint.csv` so you can reload it without re-downloading. Delete this file to force a fresh download.

- **Manual Download Option**: If automatic download fails, you can manually download `g_cpc_current.tsv.zip` from https://patentsview.org/download/data-download-tables

- **Memory Usage**: The notebook is optimized for memory efficiency. Expected usage: 2-4 GB RAM.

## Methodology Details

### Step-by-Step Process

1. **Load Clinical Trial Data**: Extract unique GVKEYs from clinical trial sponsors

2. **Map GVKEY → permno_adj**:
   - Use `discern_firm_panel_1980_2021`
   - Filter for `fyear` ∈ [2000, 2021]
   - Handle time-varying GVKEY-permno_adj mappings

3. **Map permno_adj → patent_id**:
   - Use `discern_pat_grant_1980_2021`
   - Retrieve all patents for matched firms

4. **Retrieve CPC Classifications**:
   - Query PatentsView API in batches
   - Extract CPC subsection, group, and subgroup codes
   - Handle API rate limits and errors

5. **Filter for AI Patents**:
   - Check if `cpc_subsection` starts with "G06N"
   - Flag patents as AI-related

6. **Aggregate by Firm**:
   - Group by GVKEY
   - Count total patents and AI patents per firm
   - Flag firms with ≥1 AI patent

7. **Compute Statistics**:
   - Calculate percentage of firms with AI patents
   - Generate distribution statistics
   - Export results

### Data Quality Checks

The notebook includes:
- Match rate calculation (% of clinical trial firms matched to DISCERN)
- Distribution analysis of AI patents per firm
- Top firms by AI patent count
- Temporal consistency checks

## Expected Results

Based on prior research:
- **Match Rate**: 50-80% of clinical trial firms matched to DISCERN
- **AI Patent Rate**: 5-20% of clinical trial firms have AI patents
- **Temporal Trend**: Increasing AI patent activity from 2000 to 2021

## Troubleshooting

### API Errors
If you encounter API errors:
1. Check internet connection
2. Increase `delay` parameter (e.g., 2.0 seconds)
3. Reduce `batch_size` (e.g., 50 instead of 100)
4. Load from checkpoint if previous run completed partially

### Memory Issues
If running out of memory:
1. Process patents in smaller batches
2. Close other applications
3. Use a machine with more RAM (8GB+ recommended)

### Missing Data
If DISCERN files are not found:
1. Verify paths in the notebook match your directory structure
2. Ensure DISCERN 2.0.1 files are in `../Task2 Outdated/Task2 Outdated V2/DISCERN 2_0_1/`
3. Check that .dta files are readable (may need `pip install pyreadstat`)

## Technical Notes

### CPC Code Structure
- **Format**: Section (1 letter) + Class (2 digits) + Subclass (1 letter) + Group/Subgroup
- **Example**: G06N3/04 (neural networks, architecture)
- **G06N Coverage**: All computing based on specific computational models

### DISCERN Data
- **Version**: 2.0.1
- **Source**: https://zenodo.org/records/13619821
- **Coverage**: 1980-2021
- **Purpose**: Links patents to Compustat firms via permno_adj

### PatentsView API
- **Endpoint**: https://api.patentsview.org/patents/query
- **Rate Limit**: ~60 requests/minute (adjust as needed)
- **Documentation**: https://patentsview.org/apis/api-query-language

## References

1. **DISCERN**: Arora, A., Belenzon, S., & Sheer, L. (2021). "Matching patents to Compustat firms, 1980-2015: Dynamic reassignment, name changes, and ownership structures." Research Policy, 50(5), 104217.

2. **CPC Classification**: Cooperative Patent Classification (CPC), World Intellectual Property Organization (WIPO) and European Patent Office (EPO).

3. **PatentsView**: United States Patent and Trademark Office (USPTO), PatentsView platform.

## Contact

For questions or issues with this analysis, please refer to the documentation or contact the research team.

---

**Last Updated**: March 6, 2026
**Notebook Version**: 1.0
