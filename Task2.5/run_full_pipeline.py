#!/usr/bin/env python3
"""
PubMed Linkage Pipeline - Full Dataset Processing
Process all 9,428 clinical trials with checkpoint functionality
"""

import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote
import json
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = '03780e250434b347f670b6995eaa0d524508'
BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
RATE_LIMIT_DELAY = 0.11  # 10 requests/second with API key (0.1s + buffer)
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CHECKPOINT_INTERVAL = 50  # Save progress every 50 NCT IDs

# File paths
INPUT_FILE = '../Task2/clinical_trial_sample (1).csv'
OUTPUT_FILE = 'nct_pubmed_linkage.csv'
CHECKPOINT_FILE = 'checkpoint_results.csv'

# AI Keywords for detection
AI_KEYWORDS = [
    # Core AI Terms
    'artificial intelligence', 'machine learning', 'deep learning',
    'neural network', 'convolutional neural network', 'cnn',
    'recurrent neural network', 'rnn', 'transformer',

    # AI Techniques
    'random forest', 'support vector machine', 'svm',
    'gradient boosting', 'xgboost', 'decision tree',
    'k-nearest neighbor', 'knn', 'naive bayes',
    'ensemble learning', 'supervised learning', 'unsupervised learning',
    'reinforcement learning', 'transfer learning',

    # Deep Learning Architectures
    'lstm', 'gru', 'attention mechanism', 'bert', 'gpt',
    'resnet', 'u-net', 'gan', 'generative adversarial network',
    'autoencoder', 'variational autoencoder', 'vae',

    # AI Applications in Healthcare/Drug Development
    'computer vision', 'natural language processing', 'nlp',
    'image recognition', 'object detection', 'semantic segmentation',
    'drug discovery ai', 'ai-driven', 'ai-powered', 'ai-enabled',
    'predictive modeling', 'feature extraction', 'dimensionality reduction',

    # Specific Algorithms
    'logistic regression', 'linear regression', 'pca',
    'clustering', 'classification algorithm', 'regression model'
]

def search_pubmed_by_nct(nct_id: str, api_key: str = API_KEY) -> List[str]:
    """
    Search PubMed for publications referencing a specific NCT ID.
    """
    url = f"{BASE_URL}esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': nct_id,
        'retmode': 'json',
        'retmax': 100,
        'api_key': api_key
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])

            time.sleep(RATE_LIMIT_DELAY)
            return pmids

        except Exception as e:
            logger.warning(f"ESearch attempt {attempt + 1} failed for {nct_id}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"ESearch failed for {nct_id} after {MAX_RETRIES} attempts")
                return []

    return []

def get_publication_metadata(pmids: List[str], api_key: str = API_KEY) -> Dict[str, Dict]:
    """
    Retrieve publication metadata (year, journal, title) for a list of PMIDs.
    """
    if not pmids:
        return {}

    url = f"{BASE_URL}esummary.fcgi"
    params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'json',
        'api_key': api_key
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            result = data.get('result', {})

            metadata = {}
            for pmid in pmids:
                if pmid in result:
                    pub_data = result[pmid]

                    # Extract year from pubdate
                    pubdate = pub_data.get('pubdate', '')
                    year = pubdate.split()[0] if pubdate else ''

                    metadata[pmid] = {
                        'year': year,
                        'journal': pub_data.get('source', ''),
                        'title': pub_data.get('title', '')
                    }

            time.sleep(RATE_LIMIT_DELAY)
            return metadata

        except Exception as e:
            logger.warning(f"ESummary attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"ESummary failed after {MAX_RETRIES} attempts")
                return {}

    return {}

def get_publication_abstracts(pmids: List[str], api_key: str = API_KEY) -> Dict[str, str]:
    """
    Retrieve abstracts for a list of PMIDs.
    """
    if not pmids:
        return {}

    url = f"{BASE_URL}efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml',
        'rettype': 'abstract',
        'api_key': api_key
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)

            abstracts = {}
            for article in root.findall('.//PubmedArticle'):
                # Get PMID
                pmid_elem = article.find('.//PMID')
                if pmid_elem is not None:
                    pmid = pmid_elem.text

                    # Get abstract text
                    abstract_texts = []
                    for abstract_elem in article.findall('.//AbstractText'):
                        if abstract_elem.text:
                            abstract_texts.append(abstract_elem.text)

                    abstracts[pmid] = ' '.join(abstract_texts) if abstract_texts else ''

            time.sleep(RATE_LIMIT_DELAY)
            return abstracts

        except Exception as e:
            logger.warning(f"EFetch attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"EFetch failed after {MAX_RETRIES} attempts")
                return {}

    return {}

def detect_ai_reference(title: str, abstract: str, keywords: List[str] = AI_KEYWORDS) -> bool:
    """
    Detect AI references in publication title and abstract using keyword matching.
    """
    # Combine title and abstract, convert to lowercase
    combined_text = (str(title) + ' ' + str(abstract)).lower()

    # Check for keyword matches
    for keyword in keywords:
        if keyword.lower() in combined_text:
            return True

    return False

def process_nct_id(nct_id: str) -> List[Dict]:
    """
    Process a single NCT ID: search PubMed, retrieve metadata and abstracts,
    detect AI references.
    """
    results = []

    # Step 1: Search for PMIDs
    pmids = search_pubmed_by_nct(nct_id)

    if not pmids:
        # No publications found - return row with empty values
        results.append({
            'nct_id': nct_id,
            'pmid_from_pubmed_search': '',
            'publication_year': '',
            'journal': '',
            'ai_reference_indicator': ''
        })
        return results

    # Step 2: Get metadata (year, journal, title)
    metadata = get_publication_metadata(pmids)

    # Step 3: Get abstracts
    abstracts = get_publication_abstracts(pmids)

    # Step 4: Process each publication
    for pmid in pmids:
        meta = metadata.get(pmid, {})
        abstract = abstracts.get(pmid, '')
        title = meta.get('title', '')

        # Detect AI reference
        ai_detected = detect_ai_reference(title, abstract)

        results.append({
            'nct_id': nct_id,
            'pmid_from_pubmed_search': pmid,
            'publication_year': meta.get('year', ''),
            'journal': meta.get('journal', ''),
            'ai_reference_indicator': ai_detected
        })

    return results

def process_nct_batch(nct_ids: List[str], checkpoint_file: str = CHECKPOINT_FILE) -> pd.DataFrame:
    """
    Process a batch of NCT IDs with checkpoint saving.
    """
    all_results = []

    # Check if checkpoint exists
    start_idx = 0
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        checkpoint_df = pd.read_csv(checkpoint_file)
        all_results = checkpoint_df.to_dict('records')

        # Find where to resume
        processed_nct_ids = set(checkpoint_df['nct_id'].unique())
        for i, nct_id in enumerate(nct_ids):
            if nct_id not in processed_nct_ids:
                start_idx = i
                break
        else:
            start_idx = len(nct_ids)  # All processed

        logger.info(f"Resuming from NCT ID index {start_idx}")

    # Process NCT IDs
    for i, nct_id in enumerate(tqdm(nct_ids[start_idx:], desc="Processing NCT IDs", initial=start_idx, total=len(nct_ids))):
        try:
            results = process_nct_id(nct_id)
            all_results.extend(results)

            # Save checkpoint every CHECKPOINT_INTERVAL
            if (start_idx + i + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_df = pd.DataFrame(all_results)
                checkpoint_df.to_csv(checkpoint_file, index=False)
                logger.info(f"Checkpoint saved at NCT ID {start_idx + i + 1}/{len(nct_ids)}")

        except Exception as e:
            logger.error(f"Error processing {nct_id}: {e}")
            # Add error row
            all_results.append({
                'nct_id': nct_id,
                'pmid_from_pubmed_search': '',
                'publication_year': '',
                'journal': '',
                'ai_reference_indicator': ''
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save final checkpoint
    df.to_csv(checkpoint_file, index=False)
    logger.info("Final checkpoint saved")

    return df

def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("PubMed Linkage Pipeline - Full Dataset Processing")
    logger.info("="*60)

    # Load clinical trial dataset
    logger.info(f"Loading input file: {INPUT_FILE}")
    df_clinical_trials = pd.read_csv(INPUT_FILE)
    logger.info(f"Total NCT IDs in dataset: {len(df_clinical_trials)}")

    # Get all NCT IDs
    nct_ids = df_clinical_trials['nct_id'].tolist()
    logger.info(f"Processing all {len(nct_ids)} NCT IDs")

    # Process all NCT IDs
    logger.info("Starting PubMed linkage pipeline...")
    df_results = process_nct_batch(nct_ids)
    logger.info("Pipeline completed!")

    # Save final results
    df_results.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # Calculate summary statistics
    logger.info("="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)

    total_nct_ids = df_results['nct_id'].nunique()
    logger.info(f"Total NCT IDs processed: {total_nct_ids}")

    nct_with_pubs = df_results[df_results['pmid_from_pubmed_search'] != '']['nct_id'].nunique()
    logger.info(f"NCT IDs with at least one publication: {nct_with_pubs} ({nct_with_pubs/total_nct_ids*100:.1f}%)")

    nct_without_pubs = total_nct_ids - nct_with_pubs
    logger.info(f"NCT IDs without publications: {nct_without_pubs} ({nct_without_pubs/total_nct_ids*100:.1f}%)")

    total_publications = df_results[df_results['pmid_from_pubmed_search'] != ''].shape[0]
    logger.info(f"Total publications found: {total_publications}")

    if nct_with_pubs > 0:
        avg_pubs = total_publications / nct_with_pubs
        logger.info(f"Average publications per NCT ID (with pubs): {avg_pubs:.2f}")

    ai_publications = df_results[df_results['ai_reference_indicator'] == True].shape[0]
    if total_publications > 0:
        logger.info(f"Publications with AI references: {ai_publications} ({ai_publications/total_publications*100:.1f}%)")
    else:
        logger.info(f"Publications with AI references: {ai_publications}")

    nct_with_ai = df_results[df_results['ai_reference_indicator'] == True]['nct_id'].nunique()
    logger.info(f"NCT IDs with at least one AI publication: {nct_with_ai} ({nct_with_ai/total_nct_ids*100:.1f}%)")

    logger.info("="*60)
    logger.info("Processing complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
