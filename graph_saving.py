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
import seaborn as sns
import thresholds_utils
import copy
import thresholds_utils as tu
import time
import pandas as pd

# Load data

basepath = './'
basepath_data = os.path.join(basepath, 'data/data_combinada.npy')
basepath_final = os.path.join(basepath, 'data/data540/')

data = np.load(basepath_data)
print(f'Data succesfully loaded: {data.shape}')

scan_types = ["FA", "GM", "RS"]

# GUARDEM ELS ORIGINALS SEPARATS
for i in range(data.shape[3]):
    np.save(f'data/data540/{scan_types[i]}_original.npy', data[:,:,:,i])

# Thresholding final values:
#FA_threshold = [0.45]
#GM_threshold = [0.575]
#RS_threshold = [0.1]

# Sparsification final values:
#FA_sparsification_fraction = [0.2]
#GM_sparsification_fraction = [0.4]
#RS_sparsification_fraction = [0.1]

# Saliency -> ??

methods = ['threshold']
final_thresholds = [[0.45], [0.575], [0.1]]

thr_values = [final_thresholds]


# Define a list of methods and corresponding processing functions
method_functions = [
    (methods[0], utils_thresholds.calculate_thresholded_graphs)
    # (methods[1], utils_sparsification.calculate_sparsified_graphs)
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



# ░▄▀▀▒▄▀▄░█▒░░█▒██▀░█▄░█░▄▀▀░▀▄▀
# ▒▄██░█▀█▒█▄▄░█░█▄▄░█▒▀█░▀▄▄░▒█▒


def SPT_weight(g, matrix, k):
    #distàncies i camins

    path = nx.single_source_dijkstra_path(g, source=k)
    #print(path)

    del path[k] #eliminem per exemple --> 0: [0], ja que no ens serveix d'absolutament res saber que el camí mes curt a 0 és ell mateix.

    adj = np.zeros((g.number_of_nodes(), g.number_of_nodes()))
    for target in path.values():
        for n in target[1:]:        #doble for que sembla ineficient pero cada target té màxim 3,4,5 valors
            if matrix[k][n] > 0:
                adj[k][n] += 1
                adj[n][k] += 1
    
    return adj

"""es pot estalviar molt mes temps passant-li els grafs ja creats, però aixi se li pot passar la matriu de data directament"""

def salience_weight(matrix, data_type = 0):               #0,1,2 segons el tipus de dada que volem 
    
    if data_type != 3:    
        matrix = matrix[:, :, data_type]        #agafem un dels 3

    num_nodes = matrix.shape[0]             #assumim que la matriu és quadrada
    S = np.zeros((num_nodes, num_nodes))

    matrix = np.abs(1-matrix)               #!!! fem la inversa dels pesos per a que dijkstra
    matrix[matrix == 1] = 0                 #passem els 1 a 0 pq no existeixin els camins que inicialment no hi eren
 
    g = nx.from_numpy_array(matrix)         #convertim a graf

    for node in range(num_nodes):
        S = S + SPT_weight(g, matrix, node)

    S = 1.*S/num_nodes
    return S   

def delete_edges_numperc(graf, type_delete=str, percent=50, num_edges=1000, heatmap=False, weight_name='weight'):
    
    """type_delete pot ser o 'per' de percentatge, o 'num' de numero d'arestes """

    weights = np.array([w[weight_name] for (u, v, w) in graf.edges(data=True)]) # llista de pesos (agafant els valors de weight de cada edge)
    #print('pesos', weights)
    
    if type_delete == 'per':

        threshold_value = np.percentile(weights, percent)
        
        # pesos + baixos que el llindar OUT
        edges_to_remove = [(u, v) for (u, v, w) in graf.edges(data=True) if w[weight_name] < threshold_value]
        graf.remove_edges_from(edges_to_remove)

    elif type_delete == 'num':

        edges = [(u, v) for u, v, w in weights if w[weight_name] < threshold_value]
        edges_to_remove = sorted(edges, key = lambda x: x[2], reverse=True) #agafem el [2]--> weight
        graf.remove_edges_from(edges_to_remove[num_edges:]) # em quedo amb les X primeres

    else: print('type_delete invàlid')
    if heatmap == True:
        g_mat = nx.to_numpy_array(graf)
        sns.heatmap(g_mat, cmap='viridis')

    return graf

def kruskal(g, weight='weight'):
    # invertim pq les mes grans es tornin les mes petites
    for u, v, w in g.edges(data=True):
        w[weight] = -w[weight]
    
    kr = nx.minimum_spanning_tree(g, weight='weight')
    
    # desfem els canvis
    for u, v, w in kr.edges(data=True):
        w[weight] = -w[weight]
    
    return kr

def add_edges_until_connected(G, pes='weight', comp=1):
    gx = nx.Graph()
    gx.add_nodes_from(G.nodes())
    
    # print('Arestes originals:', len(G.edges()), G.edges(data=True))

    # Ordenar les arestes per pes
    arestes = sorted(G.edges(data=True), key=lambda x: x[2].get(pes, 0), reverse=True)
    
    # Comprovació inicial
    if not arestes:
        print("No hi ha arestes per afegir.")
        return gx

    cond = list(nx.connected_components(gx))

    while len(cond) > comp:
        if not arestes:
            print("Les arestes s'han buidat abans de completar la connexió.")
            break

        add = arestes.pop(0)
        # print(f"Afegint aresta: {add}")
        gx.add_edge(add[0], add[1], **add[2])  # Afegir l'aresta amb el seu pes

        cond = list(nx.connected_components(gx))
        # print('Condicions actuals:', cond)
    
    return gx

    # ordenar arestes per pes
    # anar afegint arestes fins que connected sigui =1

    
def original_edges(g, data, patient_num, scan_type):

    """
    g: graph
    data: numpy array with all of the data
    patient_num: num patient (0, len(data))
    scan_type: 0,1,2
    """
    for edge in g.edges():
        g[edge[0]][edge[1]]['weight'] = data[patient_num][edge[0]][edge[1]][scan_type]


n = data.shape[1]
diag_mask = np.eye(n, n, dtype = bool)

new_data = []

for matrix in data:

    matrix[diag_mask, 1] = 0 
    matrix[diag_mask, 2] = 0 

    new_data.append(matrix)

data = copy.deepcopy(new_data)

for i in range(len(data)):
    data[i][:,:,2] = np.abs(matrix[:,:,2])

# print(data[0],'\n','canvi matriu',data[1])

### MAIN FUNCTION with KRUSKAL + thresholds ###

def sparsify_percentile(mat, percentile=50):
    weights = mat[mat > 0]
    thr = np.percentile(weights, percentile)
    return thr


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
            thr = thresholds_utils.graph_sparsification(sw, fraction=percentil_RS)
            thr = nx.to_numpy_array(thr)
            sw = salience_weight(thr, data_type=3)
        else:
            sw = salience_weight(data[patient], data_type=i)  # FA o GM

        g_prima_W = nx.from_numpy_array(sw)

        # Kruskal
        g_prima_W_kr = kruskal(g_prima_W)
        original_edges(g_prima_W_kr, data, patient, i)
        adj_kr = nx.to_numpy_array(g_prima_W_kr)
        llista_kr.append(adj_kr)

        # Add edges until connected (kruskal_fals)
        graf_original_comp = len(list(nx.connected_components(nx.from_numpy_array(data[patient][:, :, i]))))
        g_prima_W_krfals = add_edges_until_connected(g_prima_W, comp=graf_original_comp)
        original_edges(g_prima_W_krfals, data, patient, i)
        adj_krfals = nx.to_numpy_array(g_prima_W_krfals)
        llista_krfals.append(adj_krfals)

    # Desa els resultats de cada tipus de graf
    np.save(f'./data/data540/{datatype[i]}_salience_kr.npy', np.array(llista_kr, dtype=np.float64))
    np.save(f'./data/data540/{datatype[i]}_salience_krfals.npy', np.array(llista_krfals, dtype=np.float64))

