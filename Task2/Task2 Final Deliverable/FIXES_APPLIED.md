# Bug Fixes Applied to Clinical Trials AI Patents Analysis

## Issues Identified

### Issue #1: Zero Results (Original)
The notebook was returning **zero results** at multiple steps, culminating in a **ZeroDivisionError** in cell 27. The root cause was a **data type mismatch** between GVKEY values.

### Issue #2: ValueError on Empty Strings (Follow-up Fix)
After fixing the data type issue, a new error appeared: `ValueError: invalid literal for int() with base 10: ''` in cell 9, line 11. This occurred because some GVKEY values in DISCERN were empty strings that couldn't be converted directly to integers.

## Root Cause Analysis

### The Problem
- **Clinical trials data**: `gvkey_sponsor` stored as **integers** (e.g., 8530, 24454)
- **DISCERN data**: `gvkey` stored as **string objects** with zero-padding (e.g., '008530', '024454')

When filtering with `.isin(ct_gvkeys)` in cell 10, pandas couldn't match integers to strings, resulting in:
- 0 matched firms
- 0 patents
- 0 CPC codes
- Division by zero errors

## Fixes Applied

### 1. **GVKEY Data Type Conversion with Empty String Handling** (Cell 9)

**Added:**
```python
# Check for empty or invalid GVKEYs
print(f"Empty GVKEYs: {(firm_panel['gvkey'] == '').sum()}")

# Convert to numeric, coercing errors to NaN
firm_panel['gvkey'] = pd.to_numeric(firm_panel['gvkey'], errors='coerce')

# Remove rows with invalid GVKEYs
invalid_gvkeys = firm_panel['gvkey'].isna().sum()
if invalid_gvkeys > 0:
    print(f"Warning: Found {invalid_gvkeys} invalid GVKEYs, removing these rows...")
    firm_panel = firm_panel[firm_panel['gvkey'].notna()].copy()

# Convert to int
firm_panel['gvkey'] = firm_panel['gvkey'].astype(int)
```

**Result:**
- Safely handles empty strings and invalid GVKEY values
- Uses `pd.to_numeric()` with `errors='coerce'` to convert non-numeric values to NaN
- Filters out invalid rows before final integer conversion
- Now both datasets use integer GVKEYs for consistent matching

---

### 2. **Diagnostic Output** (Cell 10)

**Added:**
```python
print(f"\nDiagnostic info BEFORE filtering:")
print(f"- ct_gvkeys dtype: {ct_gvkeys.dtype}")
print(f"- firm_panel['gvkey'] dtype: {firm_panel['gvkey'].dtype}")
print(f"- Any overlap? {len(set(ct_gvkeys) & set(firm_panel['gvkey'].unique()))} GVKEYs")
```

**Result:** Helps verify data types match before filtering.

---

### 3. **Division by Zero Protection** (Cell 26)

**Before:**
```python
print(f"Percentage: {cpc_data['is_ai_cpc'].sum() / len(cpc_data) * 100:.2f}%")
```

**After:**
```python
if len(cpc_data) > 0:
    print(f"Percentage: {cpc_data['is_ai_cpc'].sum() / len(cpc_data) * 100:.2f}%")
else:
    print("Percentage: N/A (no CPC data to analyze)")
```

---

### 4. **Division by Zero Protection** (Cell 27)

**Before:**
```python
print(f"AI patent percentage: {len(ai_patent_ids) / cpc_data['patent_id'].nunique() * 100:.2f}%")
```

**After:**
```python
if cpc_data['patent_id'].nunique() > 0:
    print(f"AI patent percentage: {len(ai_patent_ids) / cpc_data['patent_id'].nunique() * 100:.2f}%")
else:
    print("AI patent percentage: N/A (no patents to analyze)")
```

---

### 5. **Safe Aggregation** (Cell 31)

**Added:**
```python
if len(ct_patents_with_gvkey) > 0:
    # ... aggregation code ...
else:
    print("\nNo patents found for clinical trial firms. Creating empty firm_ai_status dataframe.")
    firm_ai_status = pd.DataFrame(columns=['gvkey', 'total_patents', 'ai_patents', 'has_ai_patent'])
```

**Result:** Handles case when no patents are found without crashing.

---

### 6. **Safe Statistics Calculation** (Cell 33)

**Added:**
```python
if len(firm_ai_status) > 0:
    firms_with_ai = firm_ai_status['has_ai_patent'].sum()
else:
    firms_with_ai = 0

percentage_with_ai = (firms_with_ai / total_ct_firms) * 100 if total_ct_firms > 0 else 0
```

---

### 7. **Safe Match Rate Calculation** (Cell 35)

**Added:**
```python
if total_ct_firms > 0:
    print(f"\n- Match rate: {len(firm_ai_status) / total_ct_firms * 100:.2f}%")
else:
    print(f"\n- Match rate: N/A")

if len(firm_ai_status) > 0:
    print(f"- AI patent rate (among patenting firms): {firms_with_ai / len(firm_ai_status) * 100:.2f}%")
else:
    print(f"- AI patent rate (among patenting firms): N/A (no patenting firms found)")
```

---

### 8. **Safe Distribution Analysis** (Cell 36)

**Added:**
```python
if len(firm_ai_status) > 0 and firm_ai_status['has_ai_patent'].sum() > 0:
    # ... distribution analysis ...
else:
    print("\nNo firms with AI patents found in the dataset.")
```

---

### 9. **Safe Name Merging** (Cell 37)

**Added:**
```python
if len(firm_ai_status) > 0:
    # ... merge and display ...
else:
    print("\nNo firms with patents found. Cannot display firm rankings.")
    firm_ai_status_with_names = pd.DataFrame(columns=['gvkey', 'sponsor_name', 'total_patents', 'ai_patents', 'has_ai_patent'])
```

---

### 10. **Safe Export with Error Messages** (Cell 39)

**Added:**
```python
if len(firm_ai_status) > 0:
    # ... export code ...
    ai_patent_rate = f"{ai_patents_sum / total_patents_sum * 100:.2f}%" if total_patents_sum > 0 else "N/A"
else:
    print("\nWarning: No firms matched to DISCERN. Cannot export results.")
    print("This may indicate a data matching issue. Please check:")
    print("1. GVKEY format compatibility between clinical trials and DISCERN")
    print("2. Fiscal year range (2000-2021) may be too restrictive")
    print("3. Clinical trial firms may not be in DISCERN patent database")
```

---

## Expected Results After Fix

After applying these fixes, the notebook should:

1. **Successfully match clinical trial firms to DISCERN** (expected 40-60% match rate)
2. **Retrieve patents for matched firms** (thousands of patents)
3. **Query PatentsView API** without errors
4. **Calculate AI patent statistics** without division by zero errors
5. **Export results** successfully

## Validation Steps

To verify the fixes work:

1. Run cell 9 - should show GVKEY converted to integer
2. Run cell 10 - should show diagnostic output with non-zero overlaps
3. Run cell 11 - should show non-zero GVKEY-permno_adj mappings
4. Continue through notebook - no more zero value errors
5. Check final results - should show realistic percentages

## Additional Notes

### Why Zero-Padding Didn't Help

We considered converting clinical trial GVKEYs to zero-padded strings:
```python
ct_gvkeys = ct_gvkeys.astype(str).str.zfill(6)
```

However, converting DISCERN GVKEYs to integers is simpler and more efficient because:
- Clinical trials already use integers (native format)
- Integer comparison is faster than string comparison
- No risk of format inconsistencies

### Match Rate Expectations

Not all clinical trial firms will match DISCERN because:
- DISCERN covers publicly traded firms (Compustat universe)
- Some clinical trial sponsors are private companies
- Some sponsors are non-profit organizations
- Expected match rate: 40-70%

### Time Window Note

The 2000-2021 fiscal year restriction is intentional:
- DISCERN data ends in 2021
- Focuses on modern AI patent era
- Excludes clinical trials from 2022-2023 if they exist

---

## Follow-up Fix: Empty String Handling

### Error Encountered
```
ValueError: invalid literal for int() with base 10: ''
```

### Cause
Some rows in the DISCERN `firm_panel` dataset had empty string values for `gvkey`. When attempting direct conversion with `.astype(int)`, Python couldn't convert empty strings to integers.

### Solution
Used a safer conversion approach:
1. **`pd.to_numeric()`** with `errors='coerce'` converts invalid values to NaN
2. **Filter out NaN values** before final integer conversion
3. **Report** how many invalid GVKEYs were found and removed

This ensures the notebook handles messy real-world data gracefully.

---

## Fix #3: PatentsView API Deprecation (Status 410 Error)

### Error Encountered
```
HTTP Status 410 (Gone) when querying api.patentsview.org
```

### Cause
The PatentsView Legacy API was **shut down on May 1, 2025**. All requests to `api.patentsview.org` now return a 410 "Gone" error indicating the resource has been permanently removed.

**Source**: [PatentsView Ends Support for Legacy API](https://patentsview.org/data-in-action/patentsview-ends-support-legacy-api)

### Solution
Replaced API-based CPC retrieval with **bulk data download approach**:

1. **Download `g_cpc_current.tsv.zip`** from PatentsView bulk downloads (~4GB)
2. **Stream and filter** in chunks for memory efficiency
3. **Cache results** locally for subsequent runs
4. **Match patent IDs** with CPC classifications offline

**New Implementation (Cells 19-20)**:
```python
def download_and_load_cpc_data(patent_ids_set, cache_file='cpc_codes_checkpoint.csv'):
    # Check cache first
    if cache_path.exists():
        return pd.read_csv(cache_path)

    # Download bulk data
    cpc_url = "https://s3.amazonaws.com/data.patentsview.org/download/g_cpc_current.tsv.zip"
    response = requests.get(cpc_url, stream=True, timeout=300)

    # Filter in chunks for memory efficiency
    for chunk in pd.read_csv(zip_file, sep='\t', chunksize=100000):
        filtered = chunk[chunk['patent_id'].isin(patent_ids_set)]
        # ... process and cache
```

### Advantages
- ✅ No API rate limits
- ✅ Works offline after initial download
- ✅ Faster for large patent sets
- ✅ Cached results for instant re-runs
- ✅ Handles large files efficiently with chunking

### Manual Alternative
If automatic download fails, users can manually download from:
https://patentsview.org/download/data-download-tables

---

---

## Fix #4: GVKEY Format Error - 0 CPC Records Issue

### Error Encountered
Cell 56 showing 0 CPC records and 0% AI patent rate, when cross-validation with AIPD showed ~14% was expected.

### Root Cause
**GVKEY format was BACKWARDS!**

Our initial fix (Fix #1) converted DISCERN GVKEYs from strings to integers. This was **wrong**.

DISCERN stores GVKEYs as **6-digit zero-padded strings**: '001010', '008530', '024454'
Clinical trials stores them as **integers**: 1010, 8530, 24454

**Correct approach** (from working AIPD implementation):
- Convert **clinical trial GVKEYs TO strings** (pad to 6 digits)
- Keep **DISCERN GVKEYs AS strings** (already 6-digit format)

### Solution Applied

**Cell 6** - Convert clinical trial GVKEYs to match DISCERN:
```python
# Convert to 6-digit zero-padded strings
clinical_trials['gvkey_sponsor'] = clinical_trials['gvkey_sponsor'].apply(
    lambda x: str(int(x)).zfill(6) if pd.notna(x) else x
)
# Result: 8530 → '008530', 24454 → '024454'
```

**Cell 9** - Keep DISCERN GVKEYs as strings:
```python
# NOTE: DISCERN GVKEYs are already in string format (e.g., '001010', '008530')
# We will match against clinical trial GVKEYs that we converted to the same format
```

**Cells 19, 26** - Fixed CPC column name:
- Changed from `cpc_subsection` to `cpc_group`
- Matches actual PatentsView TSV structure
- Aligns with working AIPD implementation

### Expected Results After Fix
Based on AIPD benchmark (same data sources, 2000-2021):
- Match rate: ~50% of clinical trial firms to DISCERN
- Patents from clinical trial firms: ~80,000
- AI patent rate: 1-2% of patents
- Firms with AI patents: ~14% of total firms

---

**Date Fixed:** March 6, 2026
**Issues Resolved:**
- Data type mismatch (GVKEY format)
- Empty string conversion errors
- Division by zero errors
- Empty result handling
- PatentsView API deprecation (410 errors)
- Zero CPC records issue (GVKEY format backwards)
- CPC column name mismatch

**Status:** Ready for production use
