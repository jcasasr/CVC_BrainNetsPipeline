# Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.cm as cm
import thresholds_utils
import copy
import thresholds_utils as tu
import time
import pandas as pd

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

def sparsify_percentile(mat, percentile=50):
    weights = mat[mat > 0]
    thr = np.percentile(weights, percentile)
    return thr
