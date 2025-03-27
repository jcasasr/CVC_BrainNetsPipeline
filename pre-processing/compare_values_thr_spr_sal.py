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
import copy
import utils_salience

# Start time
absolut_start_time = time.time()


# Load data
basepath = './'
basepath_data = os.path.join(basepath, 'data/data.npy')
basepath_th = os.path.join(basepath, 'results/thresholded/')
basepath_spar = os.path.join(basepath, 'results/sparsified/')
basepath_salience = os.path.join(basepath, 'results/saliency/')
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
all_thr_for_node = [[0.45, 0.5], [0.55, 0.575], [0.1, 0.15]]
# Transform each number to the 'thr_' prefixed string
all_thr_for_node = [[f"thr_{x}" for x in sublist] for sublist in all_thr_for_node]


# 2. Second Method: sparsification (degree preserving)
print("------ Method 2: Degree-preserving sparsification ------")

FA_frac = [0.5, 0.2, 0.1]
GM_frac = [0.8, 0.5, 0.4]
RS_frac = [0.1, 0.05, 0.03]
all_fracs = [FA_frac, GM_frac, RS_frac]
# Transform each number to the 'spr_' prefixed string
all_fracs_for_node = [[f"spr_{x}" for x in sublist] for sublist in all_fracs]


# 3. Third Method: saliency analysis with Kruskaal algorithm
print("------ Method 3: Saliency analysis with Kruskaal algorithm ------")

data     =  np.load("./data/data.npy")
target   =  np.load("./data/target.npy")

n = data.shape[1]
diag_mask = np.eye(n, n, dtype = bool)

new_data = []

for matrix in data:

    matrix[diag_mask, 1] = 0 
    matrix[diag_mask, 2] = 0 

    new_data.append(matrix)

data = np.array(copy.deepcopy(new_data))

for i in range(len(data)):
    data[i][:,:,2] = np.abs(matrix[:,:,2])

datatype = ['FA', 'GM', 'RS']
thresholds = [10, 20, 30, 40, 50] #esborrarem 100-x de les arestes
graphs = []
percentil_RS = 0.25

for i in range(3):  # fa, gm, rs
    llista_kr = []
    llista_krfals = []

    for patient in range(len(data)):
        print(patient)

        if i == 2:  # RS necessita sparsification perquè el link salience no té sentit
            sw = data[patient][:, :, i]
            sw = nx.from_numpy_array(sw)
            thr = utils_salience.thresholds_utils.graph_sparsification(sw, fraction=percentil_RS)
            thr = nx.to_numpy_array(thr)
            sw = utils_salience.salience_weight(thr, data_type=3)
        else:
            sw = utils_salience.salience_weight(data[patient], data_type=i)  # FA o GM

        g_prima_W = nx.from_numpy_array(sw)

        # Kruskal
        g_prima_W_kr = utils_salience.kruskal(g_prima_W)
        utils_salience.original_edges(g_prima_W_kr, data, patient, i)
        adj_kr = nx.to_numpy_array(g_prima_W_kr)
        llista_kr.append(adj_kr)

        # Add edges until connected (kruskal_fals)
        graf_original_comp = len(list(nx.connected_components(nx.from_numpy_array(data[patient][:, :, i]))))
        g_prima_W_krfals = utils_salience.add_edges_until_connected(g_prima_W, comp=graf_original_comp)
        utils_salience.original_edges(g_prima_W_krfals, data, patient, i)
        adj_krfals = nx.to_numpy_array(g_prima_W_krfals)
        llista_krfals.append(adj_krfals)

    # Desa els resultats de cada tipus de graf
    np.save(os.path.join(basepath_salience, f"{datatype[i]}_salience_kr.npy"), np.array(llista_kr, dtype=np.float64))
    np.save(os.path.join(basepath_salience, f"{datatype[i]}_salience_krfals.npy"), np.array(llista_krfals, dtype=np.float64))
    

############

all_sali = ['kr', 'krfals']

salience_files = [
    "FA_salience_kr.npy", "FA_salience_krfals.npy",
    "GM_salience_kr.npy", "GM_salience_krfals.npy",
    "RS_salience_kr.npy", "RS_salience_krfals.npy"
]

# Crear la llista de llistes carregant el contingut amb np.load
salience_npys = []
for i in range(0, len(salience_files), 2):
    # Carregar els dos fitxers amb el mateix prefix
    # print(i, i+1)
    fitxer_1_path = os.path.join(basepath_salience, salience_files[i])
    fitxer_2_path = os.path.join(basepath_salience, salience_files[i+1])
    
    # Carregar els dos fitxers
    contingut_1 = np.load(fitxer_1_path)
    contingut_2 = np.load(fitxer_2_path)
    # Afegir-los com a subllista a llista_de_llistes
    salience_npys.append([contingut_1, contingut_2])


# 4. Ajuntar-ho tot
all_for_node = [['original'] + all_thr_for_node[i] + all_fracs_for_node[i] for i in range(len(scan_types))]


# 5. For each scan type
for scan_type in range(len(scan_types)):

    start_time = time.time()
    print(f"\n------ {scan_types[scan_type]} scans ------\n")
    print(scan_type)
    print(type(data))
    print(type(np.array(data)))

    graph_set = data[:,:,:,scan_type]

    print("Thresholding graphs...")
    thresholded_graphs = utils_thresholds.calculate_thresholded_graphs(graph_set, all_thresholds[scan_type], "thr")

    print("Sparsifying graphs...")
    sparsified_graphs = utils_sparsification.calculate_sparsified_graphs(graph_set, all_fracs[scan_type], "spr")

    all_graphs = thresholded_graphs | sparsified_graphs

    print("Loading saliency graphs...")
    # adding the salience calculations to the pack, as they had already been processed before
    # we just need to convert them back to graph nx type to do the calculations
    all_graphs[all_sali[0]] = [nx.from_numpy_array(npy_graph) for npy_graph in salience_npys[scan_type][0]] 
    all_graphs[all_sali[1]] = [nx.from_numpy_array(npy_graph) for npy_graph in salience_npys[scan_type][1]] 
    salience_graphs = {key: all_graphs[key] for key in all_sali if key in all_graphs}

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


    # 5.1. Graph measures
    print("------ Getting graph measures ------")

    save_path = basepath_plots + "all_" + scan_types[scan_type] + "_graph_measures_"
    original_graph_info = utils_preproc.get_number_of_nodes_and_edges(graph_set)

    graph_measures_thr = utils_preproc.plot_graph_metrics(thresholded_graphs, original_graph_info, save_path + "thr_", scan_types[scan_type])
    graph_measures_spr = utils_preproc.plot_graph_metrics(sparsified_graphs, original_graph_info, save_path + "spr_", scan_types[scan_type])
    graph_measures_sal = utils_preproc.plot_graph_metrics(salience_graphs, original_graph_info, save_path + "sal_", scan_types[scan_type])

    # Dump the dictionaries into a pickle file
    #file_path = basepath_measures + 'thr_' + scan_types[scan_type] + '_avg_graph_measures.pkl'
    #utils_preproc.save_graphs(file_path, graph_measures_thr)
    #file_path = basepath_measures + 'spr_' + scan_types[scan_type] + '_avg_graph_measures.pkl'
    #utils_preproc.save_graphs(file_path, graph_measures_spr)
    #file_path = basepath_measures + 'sal_' + scan_types[scan_type] + '_avg_graph_measures.pkl'
    #utils_preproc.save_graphs(file_path, graph_measures_sal)

    # Graph measures time
    graph_time = time.time()
    # Calculate the elapsed time
    elapsed_time = graph_time - thr_time
    # Print the elapsed time
    if elapsed_time > 60:
        print(f"Script took {elapsed_time/60:.2f} minutes to run.")
    else:
        print(f"Script took {elapsed_time:.2f} seconds to run.")


    # 5.2. Node measures
    print("------ Getting node measures ------")
    N = 10 # Define how many top nodes you want to consider

    save_path = basepath_plots + "all_" + scan_types[scan_type] + "_node_measures_"

    print("Calculating node measures...")
    all_graphs_including_originals = utils_preproc.append_original_graphs_to_threshold(all_graphs, graph_set)
    all_graphs_for_node_metrics = {key: all_graphs_including_originals[key] for key in all_for_node[scan_type]}
    node_measures = utils_preproc.calculate_node_measures_per_graph(all_graphs_for_node_metrics, None)
    node_measures_sal = utils_preproc.calculate_node_measures_per_graph(salience_graphs, 'sal')
    node_measures = node_measures | node_measures_sal

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
    utils_preproc.plot_node_metrics_per_threshold(all_nodes, all_common_nodes, avg_node_measures, save_path, scan_types[scan_type])


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

