# PubMed Linkage Implementation Plan
**NCT ID - PubMed Publication Mapping**

---

## 1. Overview

### Objective
Create a reproducible pipeline to link clinical trials (NCT IDs) from the clinical trial dataset to their associated publications in PubMed, and identify whether these publications reference AI or AI-enabled methods.

### Input Data
- **Source**: `Task2/clinical_trial_sample (1).csv`
- **Total NCT IDs**: 9,428 clinical trials
- **Key Fields**: `nct_id`, `sponsor_name`, `gvkey_sponsor`, `start_year`, `phase_number`

### Output Data
A CSV file containing:
- `nct_id`: Clinical trial identifier
- `pmid_from_pubmed_search`: PubMed ID(s) of associated publications
- `publication_year`: Year of publication
- `journal`: Journal name
- `ai_reference_indicator`: Boolean/flag indicating AI-related content in title/abstract

---

## 2. Technical Approach: E-utilities API

### API Base URL
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
```

### Primary E-utilities Endpoints

#### 2.1 ESearch - Search for Publications
**Endpoint**: `esearch.fcgi`

**Purpose**: Search PubMed for publications that reference a specific NCT ID

**Example Query**:
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?
  db=pubmed&
  term=NCT00175851&
  retmode=json&
  retmax=100
```

**Parameters**:
- `db=pubmed`: Search the PubMed database
- `term=NCT[ID]`: Search term (NCT ID)
- `retmode=json`: Return results in JSON format
- `retmax=100`: Maximum number of results to return per NCT ID

**Returns**: List of PMIDs associated with the NCT ID

#### 2.2 ESummary - Retrieve Publication Metadata
**Endpoint**: `esummary.fcgi`

**Purpose**: Get publication details (year, journal, title) for each PMID

**Example Query**:
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?
  db=pubmed&
  id=12345678,87654321&
  retmode=json
```

**Parameters**:
- `id`: Comma-separated list of PMIDs (can batch up to 200)
- `retmode=json`: Return format

**Returns**: Publication year, journal name, title, authors, etc.

#### 2.3 EFetch - Retrieve Full Publication Data
**Endpoint**: `efetch.fcgi`

**Purpose**: Get abstracts and detailed content for AI keyword analysis

**Example Query**:
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?
  db=pubmed&
  id=12345678&
  retmode=xml&
  rettype=abstract
```

**Parameters**:
- `retmode=xml`: Return XML format (easier to parse structured data)
- `rettype=abstract`: Include abstract text

**Returns**: Full article metadata including title and abstract text

### API Rate Limits and Best Practices
- **Rate Limit**: 3 requests/second (without API key), 10 requests/second (with API key)
- **Recommendation**: Register for NCBI API key (free) for better performance

- API Key created by user: 03780e250434b347f670b6995eaa0d524508 


- **Batch Processing**: Group PMID lookups (up to 200 PMIDs per ESummary/EFetch call)
- **Error Handling**: Implement retry logic with exponential backoff for failed requests
- **Politeness**: Add delays between requests to avoid overwhelming the server

---

## 3. Data Pipeline Architecture

### Step 1: Data Preparation
1. Load clinical trial dataset from `Task2/clinical_trial_sample (1).csv`
2. Extract unique NCT IDs (9,428 trials)
3. Initialize output data structure

### Step 2: PubMed Search (ESearch)
For each NCT ID:
1. Query PubMed using ESearch with NCT ID as search term
2. Retrieve list of associated PMIDs
3. Handle cases with:
   - No publications (empty result)
   - Single publication
   - Multiple publications
4. Store NCT ID → PMID mapping

### Step 3: Publication Metadata Retrieval (ESummary)
1. Batch PMIDs into groups of 200
2. For each batch, call ESummary to retrieve:
   - Publication year (`PubDate` field)
   - Journal name (`Source` field)
   - Article title (`Title` field)
3. Join metadata back to NCT ID - PMID pairs

### Step 4: Abstract Retrieval (EFetch)
1. Batch PMIDs into groups of 200
2. Retrieve full abstracts using EFetch
3. Parse XML response to extract:
   - Article title
   - Abstract text
4. Combine title + abstract for AI keyword analysis

### Step 5: AI Reference Detection
Apply chosen AI detection method (see Section 4) to identify publications that reference AI or AI-enabled methods.

### Step 6: Data Export
1. Combine all fields into final dataset
2. Export as CSV with schema:
   - `nct_id`
   - `pmid_from_pubmed_search`
   - `publication_year`
   - `journal`
   - `ai_reference_indicator`

---

## 4. AI Reference Detection Methods

### **Option 1: Rule-Based Keyword Matching** (RECOMMENDED)

#### Description
Scan publication titles and abstracts for predefined AI-related keywords using exact or fuzzy matching.

#### Implementation Approach
1. **Define AI Keyword Dictionary**:
   ```python
   ai_keywords = [
       # Core AI Terms
       'artificial intelligence', 'machine learning', 'deep learning',
       'neural network', 'convolutional neural network', 'CNN',
       'recurrent neural network', 'RNN', 'transformer',

       # AI Techniques
       'random forest', 'support vector machine', 'SVM',
       'gradient boosting', 'XGBoost', 'decision tree',
       'k-nearest neighbor', 'KNN', 'naive bayes',
       'ensemble learning', 'supervised learning', 'unsupervised learning',
       'reinforcement learning', 'transfer learning',

       # Deep Learning Architectures
       'LSTM', 'GRU', 'attention mechanism', 'BERT', 'GPT',
       'ResNet', 'U-Net', 'GAN', 'generative adversarial network',
       'autoencoder', 'variational autoencoder', 'VAE',

       # AI Applications in Healthcare/Drug Development
       'computer vision', 'natural language processing', 'NLP',
       'image recognition', 'object detection', 'semantic segmentation',
       'drug discovery AI', 'AI-driven', 'AI-powered', 'AI-enabled',
       'predictive modeling', 'feature extraction', 'dimensionality reduction',

       # Specific Algorithms
       'logistic regression', 'linear regression', 'PCA',
       'clustering', 'classification algorithm', 'regression model'
   ]
   ```

2. **Text Preprocessing**:
   - Convert text to lowercase
   - Remove special characters (optional)
   - Tokenize into words/phrases

3. **Matching Logic**:
   ```python
   def detect_ai_reference(title, abstract):
       """
       Returns True if any AI keyword is found in title or abstract
       """
       combined_text = (title + ' ' + abstract).lower()

       for keyword in ai_keywords:
           if keyword.lower() in combined_text:
               return True
       return False
   ```

4. **Output Format**:
   - Boolean flag: `True` (AI reference found) / `False` (no AI reference)
   - Optional: Store matched keywords for validation/analysis

#### Advantages
- Simple, transparent, and reproducible
- Fast execution (regex-based matching)
- Easy to audit and refine keyword list
- No external dependencies or models required
- Domain expert can review and adjust keywords

#### Disadvantages
- May miss contextual or implicit AI references
- Sensitive to exact wording (e.g., "ML" vs "machine learning")
- Could produce false positives (e.g., "artificial" in non-AI context)
- Requires manual curation of keyword list

#### Recommended Refinements
- Use case-insensitive matching
- Consider word boundaries to avoid partial matches (e.g., "art" in "artificial")
- Implement acronym expansion (e.g., match both "ML" and "machine learning")
- Add negative keywords to reduce false positives (e.g., exclude "artificial limb")

---

### **Option 2: NLP-Based Semantic Analysis**

#### Description
Use pre-trained Natural Language Processing models to identify AI-related content through semantic similarity or topic classification.

#### Implementation Approach

##### **Approach 2A: Sentence Embeddings + Similarity Matching**
1. **Use Pre-trained Model**:
   - SentenceTransformers (e.g., `all-MiniLM-L6-v2`)
   - BioBERT (optimized for biomedical text)

2. **Create AI Reference Corpus**:
   ```python
   ai_reference_texts = [
       "This study uses machine learning to predict drug efficacy",
       "Deep learning models were trained on patient data",
       "Artificial intelligence algorithms identified biomarkers",
       "Neural networks classified disease subtypes",
       # ... more representative AI-related sentences
   ]
   ```

3. **Compute Embeddings**:
   - Embed each publication's title + abstract
   - Embed AI reference corpus

4. **Similarity Calculation**:
   - Compute cosine similarity between publication embedding and AI reference embeddings
   - Set threshold (e.g., similarity > 0.65 indicates AI reference)

5. **Classification**:
   ```python
   from sentence_transformers import SentenceTransformer, util

   model = SentenceTransformer('all-MiniLM-L6-v2')

   def detect_ai_semantic(title, abstract, threshold=0.65):
       pub_text = title + ' ' + abstract
       pub_embedding = model.encode(pub_text)

       ai_embeddings = model.encode(ai_reference_texts)
       similarities = util.cos_sim(pub_embedding, ai_embeddings)

       max_similarity = similarities.max().item()
       return max_similarity > threshold
   ```

##### **Approach 2B: Zero-Shot Classification**
1. **Use Pre-trained Classifier**:
   - Hugging Face `facebook/bart-large-mnli` or `MoritzLaurer/deberta-v3-large-zeroshot-v2`

2. **Define Labels**:
   ```python
   labels = ["artificial intelligence research", "traditional research"]
   ```

3. **Classification**:
   ```python
   from transformers import pipeline

   classifier = pipeline("zero-shot-classification",
                         model="facebook/bart-large-mnli")

   def detect_ai_zeroshot(title, abstract):
       text = title + ' ' + abstract
       result = classifier(text, candidate_labels=labels)

       # Return True if "AI research" has higher score
       return result['labels'][0] == "artificial intelligence research"
   ```

#### Advantages
- Captures semantic meaning and context
- Less sensitive to exact wording or synonyms
- Can detect implicit AI references
- Leverages state-of-the-art NLP models

#### Disadvantages
- Requires additional libraries (transformers, sentence-transformers, torch)
- Slower execution time (especially for 9,428 NCT IDs × publications)
- Less transparent "black box" approach
- Requires GPU for efficient processing (optional but recommended)
- Harder to debug false positives/negatives
- May require threshold tuning and validation

#### Recommended Refinements
- Use domain-specific models (BioBERT, PubMedBERT) for biomedical text
- Validate on manually labeled sample before full deployment
- Combine with keyword matching (hybrid approach) for higher precision
- Cache embeddings to avoid recomputation

---

## 5. Implementation Considerations

### 5.1 Data Structure Design

#### Option A: One Row Per NCT-PMID Pair
```
nct_id       | pmid  | publication_year | journal      | ai_reference_indicator
-------------|-------|------------------|--------------|------------------------
NCT00175851  | 12345 | 2010            | Nature       | True
NCT00175851  | 67890 | 2011            | Science      | False
NCT00359632  | 11111 | 2012            | Cell         | True
```

**Advantages**:
- Easy to analyze individual publications
- Straightforward joins with other datasets
- Clear one-to-one relationship

**Disadvantages**:
- NCT IDs with multiple publications will have duplicate rows

#### Option B: One Row Per NCT with Aggregated Publications
```
nct_id       | pmid_list      | pub_years   | journals           | ai_count | ai_indicator
-------------|----------------|-------------|--------------------|-----------|--------------
NCT00175851  | [12345, 67890] | [2010,2011] | [Nature, Science]  | 1         | True
NCT00359632  | [11111]        | [2012]      | [Cell]             | 1         | True
```

**Advantages**:
- One row per NCT ID (matches input structure)
- Easier to merge with clinical trial dataset
- Can track publication counts

**Disadvantages**:
- More complex data structure (lists in CSV)
- Harder to analyze individual publications
- May need to unnest for certain analyses

**Recommendation**: Use **Option A** for flexibility, can always aggregate later if needed.

### 5.2 Handling Edge Cases

1. **NCT IDs with No Publications**:
   - Include row with `pmid = NULL` or `NaN`
   - Set `ai_reference_indicator = False` or `NULL`

2. **NCT IDs with Multiple Publications**:
   - Create separate row for each PMID
   - Optionally add `publication_count` field

3. **Publications with Missing Abstracts**:
   - Only search title for AI keywords
   - Flag with `abstract_available = False`

4. **API Request Failures**:
   - Log failed NCT IDs
   - Implement retry mechanism (3 attempts with exponential backoff)
   - Save checkpoint after each batch (e.g., every 500 NCT IDs)

5. **Ambiguous AI References**:
   - For keyword method: Review matched keywords for validation
   - For NLP method: Store confidence scores for threshold tuning

### 5.3 Performance Optimization

1. **Batch Processing**:
   - Process NCT IDs in batches of 100-500
   - Save intermediate results to avoid data loss
   - Use checkpoint/resume functionality

2. **Parallel Processing**:
   - Use multithreading for API calls (respect rate limits)
   - Process AI detection in parallel (CPU-bound)

3. **Caching**:
   - Cache PubMed API responses to avoid redundant calls
   - Store intermediate data (PMID lists, abstracts) for debugging

4. **Progress Tracking**:
   - Implement progress bar (tqdm)
   - Log statistics (e.g., "Processed 1000/9428 NCT IDs, found 2,345 publications")

### 5.4 Data Quality and Validation

1. **Sample Validation**:
   - Manually validate 50-100 random NCT-PMID pairs
   - Verify publication metadata accuracy
   - Check AI detection precision/recall on sample

2. **Quality Checks**:
   - Verify publication years are reasonable (>= trial start year)
   - Check for duplicate PMIDs per NCT ID
   - Validate journal names (no garbled text)

3. **Summary Statistics**:
   - Total NCT IDs with at least one publication
   - Total publications found
   - Average publications per NCT ID
   - Percentage of publications with AI references
   - Most common journals

---

## 6. Expected Output Schema

### Final CSV Structure
```csv
nct_id,pmid_from_pubmed_search,publication_year,journal,ai_reference_indicator
NCT00175851,20123456,2010,Nature Medicine,True
NCT00175851,21234567,2011,JAMA,False
NCT00359632,22345678,2012,The Lancet,True
NCT00415155,,,No publications found,
```

### Field Descriptions
- **nct_id**: Clinical trial identifier (string)
- **pmid_from_pubmed_search**: PubMed ID (integer or empty if no publications)
- **publication_year**: Year of publication (integer, YYYY format)
- **journal**: Full journal name (string)
- **ai_reference_indicator**: Boolean flag (True/False) or binary (1/0)

### Optional Additional Fields
Consider including for enhanced analysis:
- `publication_count`: Number of publications per NCT ID
- `first_publication_year`: Earliest publication year for the trial
- `time_to_publication`: Years between trial start and first publication
- `ai_keywords_matched`: List of AI keywords found (if using keyword method)
- `abstract_available`: Boolean flag indicating if abstract was retrieved

---

## 7. Implementation Workflow (Completed Now)

### Phase 1: Setup and Exploration 
1. ✓ Set up development environment
2. ✓ Install required libraries (`requests`, `pandas`, `xml.etree`, optional: `transformers`)
3. ✓ Register for NCBI API key
4. ✓ Test E-utilities API with sample NCT IDs (5-10 trials)
5. ✓ Verify data retrieval pipeline works end-to-end

### Phase 2: Core Pipeline Development 
1. ✓ Implement ESearch function (NCT ID → PMIDs)
2. ✓ Implement ESummary function (PMIDs → metadata)
3. ✓ Implement EFetch function (PMIDs → abstracts)
4. ✓ Build error handling and retry logic
5. ✓ Test on 100 NCT IDs

### Phase 3: AI Detection Implementation 
1. ✓ Choose AI detection method (keyword vs. NLP)
2. ✓ Implement detection function
3. ✓ Validate on sample publications (manual review)
4. ✓ Tune parameters (keywords or threshold)

### Phase 4: Full Dataset Processing 
1. ✓ Run pipeline on all 9,428 NCT IDs
2. ✓ Monitor progress and handle errors
3. ✓ Save intermediate checkpoints
4. ✓ Generate final CSV output

### Phase 5: Validation and Documentation 
1. ✓ Perform quality checks on output
2. ✓ Generate summary statistics
3. ✓ Manually validate sample (50-100 rows)
4. ✓ Document methodology in README
5. ✓ Create reproducible Jupyter notebook

---

## 8. Deliverables

### Primary Deliverables
1. **CSV File**: `nct_pubmed_linkage.csv`
   - Contains all required fields
   - One row per NCT-PMID pair (or one row per NCT with NULL for no publications)

2. **Jupyter Notebook**: `Task2.5_PubMed_Linkage.ipynb`
   - Complete pipeline implementation
   - Code comments and markdown explanations
   - Sample outputs and visualizations
   - Reproducible execution

3. **Documentation**: `README.md`
   - Methodology overview
   - API usage details
   - AI detection method description
   - Data dictionary
   - Summary statistics
   - Instructions to reproduce

### Optional Deliverables
1. **Summary Report**: `pubmed_linkage_summary.md`
   - Key findings (e.g., X% of trials have publications, Y% mention AI)
   - Data quality notes
   - Visualizations (publication trends, AI adoption over time)

2. **Validation Sample**: `validation_sample.csv`
   - 100 manually reviewed rows for accuracy check

---

## 9. Required Python Libraries

### Core Libraries
```python
import requests          # API calls
import pandas as pd      # Data manipulation
import time             # Rate limiting
import xml.etree.ElementTree as ET  # XML parsing
from urllib.parse import quote      # URL encoding
import json             # JSON parsing
```

### Optional (for NLP approach)
```python
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
```

### Utilities
```python
from tqdm import tqdm   # Progress bars
import logging          # Error logging
```

---

## 10. Success Criteria

1. **Coverage**: Successfully query all 9,428 NCT IDs
2. **Data Quality**:
   - Valid PMIDs and metadata for all found publications
   - No missing fields (except where publications don't exist)
   - AI detection runs on 100% of publications with abstracts
3. **Reproducibility**: Pipeline can be re-run and produces identical results
4. **Documentation**: Clear methodology that can be understood and replicated
5. **Validation**: Sample check shows >95% accuracy for AI detection

---

## 11. Next Steps After Plan Approval

Once this plan is reviewed and approved:

1. **Decision Point**: Choose AI detection method (Option 1 or Option 2)
2. **Finalize Output Schema**: Confirm CSV structure and field names
3. **Begin Implementation**: Start with Phase 1 (Setup and Exploration)
4. **Iterative Refinement**: Test on small sample, refine, then scale up

---

## Appendix: Example Code Snippets

### A. ESearch Query
```python
def search_pubmed_by_nct(nct_id, api_key=None):
    """Search PubMed for publications referencing NCT ID"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        'db': 'pubmed',
        'term': nct_id,
        'retmode': 'json',
        'retmax': 100
    }

    if api_key:
        params['api_key'] = api_key

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        return pmids
    else:
        return []
```

### B. Keyword-Based AI Detection
```python
def detect_ai_keywords(title, abstract):
    """Detect AI references using keyword matching"""
    ai_keywords = [
        'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'random forest', 'support vector machine',
        # ... add more keywords
    ]

    combined_text = (str(title) + ' ' + str(abstract)).lower()

    for keyword in ai_keywords:
        if keyword.lower() in combined_text:
            return True

    return False
```

---

**End of Plan**
