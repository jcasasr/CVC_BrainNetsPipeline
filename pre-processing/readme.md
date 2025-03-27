# Data Pre-Processing

This folder contains scripts and tools for preprocessing data before analysis.

## Overview
Data preprocessing involves cleaning and transforming raw data into a big array stored in 'data.npy'.

## Scripts
- `data cleaner.ipynb`: A script for cleaning missing or incorrect data entries and saving the data.
- `utils.py`: A script with the definitions of the used functions.

## Usage
To run the data cleaner script, please ensure the data is stored in the following way:

Subject scans from Barcelona in one folder, with each scan types in folders named FA, GM and RS. Subject scans from Naples in another folder, with each scan types in folders named FA, GM and RS. Example:

```
Your local folder
├── Barcelona
|     ├── FA
|     |    ├── scan1.csv
|     |    ├── scan2.csv
|     |    ├── ...
|     |    └── scanM.csv
|     ├── GM
|     └── RS
├── Naples
|     ├── FA
|     ├── GM
|     └── RS
└── Both
```

Also make sure to change the folder paths to your local paths to run 'data cleaner.ipynb'.

You can skip this step by just downloading the files 'data_BCN.npy', 'data_NAP.npy', 'target.npy' and 'ID_info.csv'.

One you have all the pre-processed data, you can run 'data analysis.ipynb'.


```bash
python clean_data.py --input data/raw_data.csv --output data/cleaned_data.csv
