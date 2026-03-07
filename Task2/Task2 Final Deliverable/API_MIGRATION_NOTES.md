# PatentsView API Migration: Status 410 Fix

## Problem Summary

**Error**: HTTP Status 410 (Gone) when querying PatentsView API in cell 20
**Root Cause**: PatentsView Legacy API was shut down on May 1, 2025

## What Changed

### Old Approach (DEPRECATED)
```python
# Query API for each patent batch
api_url = "https://api.patentsview.org/patents/query"
response = requests.post(api_url, json=query)
# ❌ Returns 410 - API no longer exists
```

### New Approach (CURRENT)
```python
# Download bulk CPC data once, filter locally
cpc_url = "https://s3.amazonaws.com/data.patentsview.org/download/g_cpc_current.tsv.zip"
response = requests.get(cpc_url)
# ✅ Download 4GB file, cache locally, filter for our patents
```

## Benefits of New Approach

| Feature | Old API | New Bulk Download |
|---------|---------|-------------------|
| Speed (first run) | 30-60 min | 10-30 min |
| Speed (subsequent) | 30-60 min | < 1 min (cached) |
| Rate limits | Yes (60/min) | No |
| Requires internet | Every time | Once |
| File size | N/A | ~4GB download |
| Local storage | ~50MB cache | ~4GB + filtered cache |

## Usage

### First Run
1. Downloads `g_cpc_current.tsv.zip` (~4GB) from PatentsView
2. Filters for your patent IDs
3. Caches results to `cpc_codes_checkpoint.csv`
4. Takes 10-30 minutes depending on internet speed

### Subsequent Runs
1. Loads from `cpc_codes_checkpoint.csv`
2. Takes < 1 minute
3. No internet required

### Force Re-download
```bash
# Delete cache file
rm cpc_codes_checkpoint.csv

# Re-run notebook - will download fresh data
```

## Manual Download Option

If automatic download fails:

1. Visit https://patentsview.org/download/data-download-tables
2. Download **`g_cpc_current.tsv.zip`** (Granted Patents CPC Current)
3. Save to notebook directory
4. Update cell 19 to load from local file

## Technical Details

### Memory Management
- File is processed in **100,000-row chunks**
- Only keeps rows matching your patent IDs
- Final filtered dataset is much smaller than full 4GB file

### What's Included
The bulk download contains:
- `patent_id`: Patent number (string)
- `cpc_section`: CPC section (e.g., 'G')
- `cpc_class`: CPC class (e.g., '06')
- `cpc_subclass`: CPC subclass (e.g., 'N')
- `cpc_group`: CPC group
- `cpc_subgroup`: CPC subgroup

We filter for patents where `cpc_section + cpc_class + cpc_subclass = 'G06N'`

## Migration Checklist

- [x] Removed deprecated API calls (cells 19-20)
- [x] Added bulk download function (cell 19)
- [x] Updated cell 20 to use bulk download
- [x] Added caching mechanism
- [x] Updated README with new instructions
- [x] Added manual download instructions (cell 23-24)
- [x] Tested with sample data

## Expected Output

### Cell 19
```
Function defined. Ready to download and process CPC codes.
```

### Cell 20 (First Run)
```
Starting CPC code retrieval for 12,543 patents...
This may take 10-30 minutes on first run (downloading bulk data).
Subsequent runs will be much faster (using cached data).

Downloading CPC data from PatentsView bulk downloads...
Note: This file is large (~4GB) and may take 10-30 minutes to download and process.

Downloading from: https://s3.amazonaws.com/data.patentsview.org/download/g_cpc_current.tsv.zip
Download complete. Extracting...
Extracting g_cpc_current.tsv...
Loading and filtering CPC data for 12,543 patents...
  Processed 1,000,000 rows, found 2,341 matches...
  Processed 2,000,000 rows, found 5,892 matches...
  ...

Filtered to 45,231 CPC records for 12,134 patents
Cached CPC data to cpc_codes_checkpoint.csv

CPC data retrieved: 45,231 rows
Unique patents with CPC data: 12,134
```

### Cell 20 (Subsequent Runs)
```
Starting CPC code retrieval for 12,543 patents...
Loading CPC data from cache: cpc_codes_checkpoint.csv
Loaded 45,231 CPC records from cache

CPC data retrieved: 45,231 rows
Unique patents with CPC data: 12,134
```

## Troubleshooting

### Download Fails
**Error**: `ConnectionError` or timeout
**Solution**:
- Check internet connection
- Increase timeout in cell 19: `timeout=600` (10 min)
- Or download manually (see above)

### Out of Memory
**Error**: `MemoryError` during processing
**Solution**:
- Reduce chunk size in cell 19: `chunksize=50000`
- Close other applications
- Use machine with more RAM (8GB+ recommended)

### Wrong CPC Columns
**Error**: `KeyError: 'cpc_subsection'`
**Solution**: The code auto-detects column structure and builds `cpc_subsection` from component parts

### Cache Corrupted
**Error**: Invalid data when loading cache
**Solution**:
```bash
rm cpc_codes_checkpoint.csv
# Re-run notebook
```

## References

- [PatentsView Deprecation Notice](https://patentsview.org/data-in-action/patentsview-ends-support-legacy-api)
- [PatentsView Bulk Downloads](https://patentsview.org/download/data-download-tables)
- [CPC Classification Info](https://www.cooperativepatentclassification.org/)

---

**Migration Date**: March 6, 2026
**Notebook Version**: 2.0 (Bulk Download Edition)
**Status**: ✅ Production Ready
