"""
# Description:
# This script processes CSV files containing subjects' scans for FA, GM, and RS.
# It applies different thresholds to the edge values for each scan type and
# performs sparsification to reduce the density of the data.

# Input:
# - CSV files for FA, GM, and RS scans.

# Output:
# - Processed data with thresholded graphs for each scan type.
# - Processed data with sparsified graphs for each scan type.

# Note:
# - Ensure the input CSV files follow the required structure.
"""

# Import libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import random
from collections import defaultdict
import utils_preproc
import utils_thresholds
import utils_sparsification

# Load data

basepath = './'
basepath_data = os.path.join(basepath, 'data/data.npy')
basepath_final = os.path.join(basepath, 'data/')

data = np.load(basepath_data)
print(f'Data succesfully loaded: {data.shape}')

scan_types = ["FA", "GM", "RS"]


# Thresholding final values:
#FA_threshold = [0.45]
#GM_threshold = [0.575]
#RS_threshold = [0.1]

# Sparsification final values:
#FA_sparsification_fraction = [0.2]
#GM_sparsification_fraction = [0.4]
#RS_sparsification_fraction = [0.1]


methods = ['threshold', 'sparsification']
final_thresholds = [[0.45], [0.575], [0.1]]
final_spar_fractions = [[0.2], [0.4], [0.1]]

thr_values = [final_thresholds, final_spar_fractions]


# Define a list of methods and corresponding processing functions
method_functions = [
    (methods[0], utils_thresholds.calculate_thresholded_graphs),
    (methods[1], utils_sparsification.calculate_sparsified_graphs)
]

# Loop over each method and its corresponding function
for method_index, (method_name, process_function) in enumerate(method_functions):
    print(f"method: {method_name}")

    # Loop over each scan type
    for scan_type_index in range(len(scan_types)):
        print(f"Scan type: {scan_types[scan_type_index]}")

        # Extract the graph set for the current scan type
        graph_set = data[:, :, :, scan_type_index]

        # Process the graph set with the appropriate function and threshold values
        thr_graphs = process_function(graph_set, thr_values[method_index][scan_type_index], method_name)
        
        # Get the first list of graphs
        thr_graphs_list = next(iter(thr_graphs.values()))

        # Save the processed graphs in a single .npy file
        #utils_preproc.save_graphs_as_npy(thr_graphs_list, basepath_final, f"{method_name}_{scan_types[scan_type_index]}")
        utils_preproc.save_all_graphs_as_npy(thr_graphs_list, basepath_final + scan_types[scan_type_index] + "_" + method_name + '.npy')
