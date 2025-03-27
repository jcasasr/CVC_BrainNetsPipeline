"""
# Description:
# This script processes the CSV files containing the subjects' scans for FA, GM, and RS.
# The scans include both the original data and the data after applying thresholding and salience methods.
# It generates node embeddings by calculating centrality metrics for each subject and
# incorporates additional features, such as node volumes.

# Input:
# - CSV files for FA, GM, and RS scans (original and processed with thresholding and salience).

# Output:
# - Node embeddings saved in various .csv files, which include:
#   - Centrality metrics for each node
#   - Node volumes for each subject

# Note:
# - Ensure that the input CSV files are correctly preprocessed and follow the expected format.
"""

# Import libraries
import numpy as np
import os
import csv
import pandas as pd
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import time
import utils_embeddings

# Start time
absolut_start_time = time.time()


# Load data

basepath = './'
basepath_data = os.path.join(basepath, 'data')
basepath_embeddings = os.path.join(basepath, 'node_embeddings')
basepath_node_vol = os.path.join(basepath, 'node_volumes')
# This folder contains the excel files with node volume information, which I manually retouched
basepath_node_vol_info = "./info volum nodes/"

scan_types = ["FA", "GM", "RS"]
processed_method = ["original", "salience_kr", "salience_krfals", "threshold"]#, "sparsification"]

all_filepaths = []

for i in range(len(scan_types)):
    current_scan_type = scan_types[i]
    # Create the list manually bc I want original at the beginning and spars at the end (or no spars at all)
    file_names = [current_scan_type + "_" + method + ".npy" for method in processed_method]

    # Full file paths
    file_paths = [os.path.join(basepath_data, f) for f in file_names]

    all_filepaths.append(file_paths)

# Load and stack all arrays for each scan type, along the last axis (axis=-1)
# This will give us shape (270, 76, 76, m) for each scan type (original + m thresholded versions, m = 4 or 5)
data = np.stack([np.stack([np.load(file) for file in files], axis=-1) for files in all_filepaths], axis=-2)

# Data loading time
data_time = time.time()
# Calculate the elapsed time
elapsed_time = data_time - absolut_start_time
# Print the elapsed time
if elapsed_time > 60:
    print(f"Data succesfully loaded {data.shape}, in {elapsed_time/60:.2f} minutes.")
else:
    print(f"Data succesfully loaded {data.shape}, in {elapsed_time:.2f} seconds.")


# Embeddings with node metrics

def process_subject(subject_id, scan_type_id, method_id, output_dir):
    """
    Process a single subject's data and save the metrics to a CSV file.
    """
    G = nx.from_numpy_array(data[subject_id, :, :, scan_type_id, method_id])
    metrics = utils_embeddings.calculate_metrics(G)
    subject_filename = f"{subject_id:04}_{scan_types[scan_type_id]}_{processed_method[method_id]}_embed.csv"
    utils_embeddings.save_csv_file(metrics, output_dir, subject_filename)

# Main processing function with parallel execution
def process_all_subjects(data, output_dir, max_workers=4):
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for subject_id in range(data.shape[0]):
            for scan_type_id in range(data.shape[3]):
                for method_id in range(data.shape[4]):
                    # Schedule each subject’s processing as a separate task
                    tasks.append(executor.submit(process_subject, subject_id, scan_type_id, method_id, output_dir))
        
        # Wait for all tasks to complete
        for task in tasks:
            task.result()

# Run the optimized function
process_all_subjects(data, basepath_embeddings)

# Metrics time
metrics_time = time.time()
# Calculate the elapsed time
elapsed_time = metrics_time - data_time
# Print the elapsed time
if elapsed_time > 60:
    print(f"Node embeddings succesfully saved in {elapsed_time/60:.2f} minutes.")
else:
    print(f"Node embeddings succesfully saved in {elapsed_time:.2f} seconds.")


# Embeddings with node volumes

file1 = "VOLUM_nNODES_CONTROLS.xls"
file2 = "VOLUM_nNODES_PATIENTS.xls"
file3 = "NODES_NAPLES.csv"

# Read the Excel file into a DataFrame
vol_node_BCN_contr = pd.read_excel(basepath_node_vol_info + file1, sheet_name='NODES_CONTROLS')
vol_node_BCN_pat = pd.read_excel(basepath_node_vol_info + file2, sheet_name='NODES_PATIENTS')
vol_node_NAP = pd.read_csv(basepath_node_vol_info + file3, sep=r'\s+')#, header=None)

# Load ID correspondence file
id_corr = pd.read_csv('./data/ID_corr.csv')
id_map = dict(zip(id_corr['ID'], id_corr['ID_old']))

# Map data files to labels
data_files = {
    'VOLUM_nNODES_CONTROLS': vol_node_BCN_contr,
    'VOLUM_nNODES_PATIENTS': vol_node_BCN_pat,
    'NODES_NAPLES': vol_node_NAP
}

# Initialize a variable to store the maximum value
global_max = float('-inf')

# Iterate over the files
for label, df in data_files.items():
    data_to_process = df.iloc[1:, 1:] # Slice to exclude the first row and column
    file_max = data_to_process.values.max()  # Get max value for this file
    global_max = max(global_max, file_max)  # Update global maximum

print(f"The maximum value across all files is: {global_max}")

# Iterate over each subject in the ID correspondence file
for new_id, old_name in id_map.items():
    
    # Dictionary to collect data from each file for this subject
    combined_data = {}

    # Retrieve data from each file for the subject's old name
    for label, df in data_files.items():
        # Dynamically get the first column name (ID column) in case it's different or has whitespace
        id_column = df.columns[0]  
        
        # Filter the subject row based on the dynamically identified ID column name
        subject_row = df[df[id_column] == old_name]
        
        # If the subject's data is found, add each 'val' column to combined_data
        if not subject_row.empty:
            for col in subject_row.columns[1:]:  # Skip the ID column
                if col not in combined_data:
                    combined_data[col] = []
                combined_data[col].append(subject_row[col].values[0])

    # Convert combined data to DataFrame where each column is a different data file
    combined_df = pd.DataFrame(combined_data)
    
    # Transpose to ensure each 'val' field is a row, with data sources as columns
    combined_df = combined_df.transpose()

    # Normalize each column to the [0, 1] range using Min-Max scaling
    # where min = 0 and max = max[all values of all patients]
    combined_df = (combined_df) / (global_max)

    # Save each subject's combined data to a separate CSV file
    filename = f"{new_id:04}_node_vol.csv"
    filepath = os.path.join(basepath_node_vol, filename)
    combined_df.to_csv(filepath, index=False, header=False)

# Volumes time
vol_time = time.time()
# Calculate the elapsed time
elapsed_time = vol_time - metrics_time
# Print the elapsed time
if elapsed_time > 60:
    print(f"Node volumes succesfully saved in {elapsed_time/60:.2f} minutes.")
else:
    print(f"Node volumes succesfully saved in {elapsed_time:.2f} seconds.")

print(f"Script took {(vol_time - absolut_start_time)/60:.2f}min to run 😊.")
