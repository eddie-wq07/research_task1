# Task 1: PatentsView Database Exploration

Programmatic pipeline for downloading and analyzing patent data from PatentsView using Python and DuckDB. Downloads patent data directly from S3 URLs using native Python libraries (`urllib`, `zipfile`) - no manual downloads required.

## Overview

- Downloads 4 core PatentsView tables (~2 GB for 2021 data)
- Identifies AI patents using CPC codes (G06N family)
- Enables biopharma domain filtering and firm-level analysis
- 2021 test dataset: 363,829 patents, 10,782 AI-related

## Usage

Open `task1_patentsview_exploration.ipynb` and run all cells - data downloads automatically.
