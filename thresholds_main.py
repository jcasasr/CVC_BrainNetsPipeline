# Import libraries

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import random
from collections import defaultdict
import utils_thresholds
import utils_preproc
import utils_sparsification
import pickle
import time

# Start time
absolut_start_time = time.time()


# Load data

basepath = './'
basepath_data = os.path.join(basepath, 'data/data.npy')
basepath_th = os.path.join(basepath, 'results/thresholded/')
basepath_spar = os.path.join(basepath, 'results/sparsified/')
basepath_plots = os.path.join(basepath, 'results/plots/')
basepath_measures = os.path.join(basepath, 'results/measures/')

data = np.load(basepath_data)
print(f'Data succesfully loaded: {data.shape}')

scan_types = ["FA", "GM", "RS"]

all_nodes = {i for i in range(76)}


# 1. First Method: threshold (edge filtering)
print("------ Method 1: Threshold via edge filtering ------")

# FA: Search around 0.3
FA_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# GM: Search around 0.55
GM_thresholds = [0.5, 0.55, 0.575, 0.6]
# RS: Search between 0.1 and 0.3
RS_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]

all_thresholds = [FA_thresholds, GM_thresholds, RS_thresholds]
all_thr_for_node = [[0, 0.4, 0.45, 0.5], [0, 0.5, 0.55, 0.575, 0.6], [0, 0.1, 0.15, 0.2, 0.25]]



# 2. Second Method: sparsification (degree preserving)
print("------ Method 2: Degree-preserving sparsification ------")

FA_frac = [0.5, 0.3, 0.1]
GM_frac = [0.9, 0.4]
RS_frac = [0.1, 0.05, 0.03]
all_fracs = [FA_frac, GM_frac, RS_frac]


# 3. Third Method: saliency analysis with Kruskaal algorithm
print("------ Method 3: Saliency analysis with Kruskaal algorithm ------")

aaa = ['sal', 'kkl']
all_sali = [aaa, aaa, aaa]

all_for_node = [[0, 0.4, 0.45, 0.5] + FA_frac + aaa, [0, 0.5, 0.55, 0.575, 0.6] + GM_frac + aaa, [0, 0.1, 0.15, 0.2, 0.25] + RS_frac + aaa]
all_for_node = [[0, 0.4, 0.45, 0.5] + FA_frac, [0, 0.5, 0.55, 0.575, 0.6] + GM_frac, [0, 0.1, 0.15, 0.2, 0.25] + RS_frac]



# For each scan type
for scan_type in range(len(scan_types)):

    start_time = time.time()
    print(f"\n------ {scan_types[scan_type]} scans ------\n")
    graph_set = data[:,:,:,scan_type]

    print("Thresholding graphs...")
    thresholded_graphs = utils_thresholds.calculate_thresholded_graphs(graph_set, all_thresholds[scan_type])

    print("Sparsifying graphs...")
    sparsified_graphs = utils_sparsification.calculate_sparsified_graphs(graph_set, all_fracs[scan_type])

#    print("Saliencing graphs...")
#    saliency_graphs = None
#    print("TODO: load sth")#thresholded_graphs = utils_thresholds.calculate_thresholded_graphs(graph_set, all_thresholds[scan_type])

    all_graphs = thresholded_graphs | sparsified_graphs# | saliency_graphs

    # Graph saving
    #file_path = basepath_th + 'thr_' + scan_types[scan_type] + '_thresholded_graphs.pkl'
    #utils_preproc.save_graphs(file_path, thresholded_graphs)

    # Thresholding time
    thr_time = time.time()
    # Calculate the elapsed time
    elapsed_time = thr_time - start_time
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")


    # 1.1. Graph measures
    print("------ Getting graph measures ------")

    save_path = basepath_plots + "all_" + scan_types[scan_type] + "_graph_measures_"
    original_graph_info = utils_preproc.get_number_of_nodes_and_edges(graph_set)
    graph_measures = utils_preproc.plot_graph_metrics(all_graphs, original_graph_info, save_path)

    # Dump the dictionaries into a pickle file
    #file_path = basepath_measures + 'thr_' + scan_types[scan_type] + '_avg_graph_measures.pkl'
    #utils_preproc.save_graphs(file_path, graph_measures)

    # Graph measures time
    graph_time = time.time()
    # Calculate the elapsed time
    elapsed_time = graph_time - thr_time
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")

    # 1.2. Node measures
    print("------ Getting node measures ------")
    N = 10 # Define how many top nodes you want to consider

    save_path = basepath_plots + "all_" + scan_types[scan_type] + "_node_measures_"

    print("Calculating node measures...")
    all_graphs_including_originals = utils_preproc.append_original_graphs_to_threshold(all_graphs, graph_set)
    all_graphs_for_node_metrics = {key: all_graphs_including_originals[key] for key in all_for_node[scan_type]}
    node_measures = utils_preproc.calculate_node_measures_per_graph(all_graphs_for_node_metrics, scan_types[scan_type])

    # Node measures time
    node_time = time.time()
    # Calculate the elapsed time
    elapsed_time = node_time - graph_time
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")

    print("Saving node measures...")
    # Dump the dictionaries into a pickle file
    file_path = basepath_measures + 'all_' + scan_types[scan_type] + '_node_measures.pkl'
    utils_preproc.save_graphs(file_path, node_measures)

    print("Averaging...")
    avg_node_measures = utils_preproc.calculate_average_node_measures_per_threshold(node_measures)
    common_nodes = utils_preproc.find_common_nodes_across_metrics(avg_node_measures, N)
    all_common_nodes = set.intersection(*common_nodes.values())

    print("Plotting!")
    utils_preproc.plot_node_metrics_per_threshold(all_nodes, all_common_nodes, avg_node_measures, save_path)

    # End time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")

# Total time
absolut_end_time = time.time()
# Calculate the elapsed time
elapsed_time = absolut_end_time - absolut_start_time
print(f"--- Thresholding script took {elapsed_time/60:.2f} minutes to run.")


'''

# Start time
start_time_spar = time.time()

# For each scan type
for scan_type in range(len(scan_types)):

    start_time = time.time()
    print(f"\n------ {scan_types[scan_type]} scans ------\n")
    graph_set = data[:,:,:,scan_type]

    print("Sparsifying graphs...")
    sparsified_graphs = utils_sparsification.calculate_sparsified_graphs(graph_set, all_fracs[scan_type])

    # ATENCIÓ: de moment els graphs els guardo en un pickle,
    # que conté tots els grafs per tots els thresholds; si de
    # cas es pot mirar més endavant de gruardar-los en .npy com
    # el data.npy per cada threshold: data_02.npy, etc.
    file_path = basepath_th + method + scan_types[scan_type] + '_sparsified_graphs.pkl'
    utils_preproc.save_graphs(file_path, sparsified_graphs)

    # Thresholding time
    thr_time = time.time()
    # Calculate the elapsed time
    elapsed_time = thr_time - start_time_spar
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")


    # 2.1. Graph measures
    print("------ Getting graph measures ------")

    save_path = basepath_plots + method + scan_types[scan_type] + "_graph_measures_"
    original_graph_info = utils_preproc.get_number_of_nodes_and_edges(graph_set)
    graph_measures = utils_preproc.plot_graph_metrics(sparsified_graphs, original_graph_info, save_path)

    # Dump the dictionaries into a pickle file
    file_path = basepath_measures + method + scan_types[scan_type] + '_avg_graph_measures.pkl'
    utils_preproc.save_graphs(file_path, graph_measures)

    # Graph measures time
    graph_time = time.time()
    # Calculate the elapsed time
    elapsed_time = graph_time - thr_time
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")

    # 2.2. Node measures
    print("------ Getting node measures ------")
    N = 10 # Define how many top nodes you want to consider

    save_path = basepath_plots + "spr_" + scan_types[scan_type] + "_node_measures_"

    print("Calculating node measures...")
    sparsified_graphs_including_originals = utils_preproc.append_original_graphs_to_threshold(sparsified_graphs, graph_set)
    node_measures = utils_preproc.calculate_node_measures_per_graph(sparsified_graphs_including_originals, scan_types[scan_type])

    # Node measures time
    node_time = time.time()
    # Calculate the elapsed time
    elapsed_time = node_time - graph_time
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")

    print("Saving node measures...")
    # Dump the dictionaries into a pickle file
    file_path = basepath_measures + method + scan_types[scan_type] + '_node_measures.pkl'
    utils_preproc.save_graphs(file_path, node_measures)

    print("Averaging...")
    avg_node_measures = utils_preproc.calculate_average_node_measures_per_threshold(node_measures)
    common_nodes = utils_preproc.find_common_nodes_across_metrics(avg_node_measures, N)
    all_common_nodes = set.intersection(*common_nodes.values())

    print("Plotting!")
    utils_preproc.plot_node_metrics_per_threshold(all_nodes, all_common_nodes, avg_node_measures, save_path)

    # End time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")

# Total time
absolut_end_time = time.time()
# Calculate the elapsed time
elapsed_time = absolut_end_time - start_time_spar
print(f"--- Sparsification script took {elapsed_time/60:.2f} minutes to run.")

# Total script time
elapsed_time = absolut_end_time - absolut_start_time
print(f"FINALLY: Full script took {elapsed_time/60:.2f} minutes to run.")
'''