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
    
    print('Arestes originals:', len(G.edges()), G.edges(data=True))

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

### INICI DEL CODI ###
t0 = time.time()

data     =  np.load("./data/data.npy")
data_NAP =  np.load("./data/data_NAP.npy")
data_BCN =  np.load("./data/data_BCN.npy")
target   =  np.load("./data/target.npy")

# print(data)


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

for patient in range(len(data)):
    print(patient)
    for i in range(3): #fa, gm, rs
        
        if i == 2: #fem sparcification de la RS pq no té cap mena de sentit fer link salience per un graf amb diametre 1
            sw = data[patient][:,:,i]
            sw = nx.from_numpy_array(sw)
            thr = thresholds_utils.graph_sparsification(sw, fraction=percentil_RS)
            thr = nx.to_numpy_array(thr)
            sw = salience_weight(thr, data_type=3)

            # print('a', sw.size)
        else: sw = salience_weight(data[patient], data_type=i) #aqui es pot posar data_type=0,1,2 segons si volem mirar FA GM o RS, predeterminat és 0
        
        g_prima_W = nx.from_numpy_array(sw)


        # kruskal_fals = add_edges_until_connected(g_prima_W) -----> no torna valors esperats encara, els weights creats per salience son extremadament iguals, no tenen sentit. especialment a RS
        g_prima_W = kruskal(g_prima_W)

        # original_edges(kruskal_fals, data, patient, i)
        original_edges(g_prima_W, data, patient, i)

        adj = nx.to_numpy_array(g_prima_W)
        np.save(f'./graphs/{datatype[i]}_{patient:04}_salience_kr_{target[patient]}.npy', adj.astype(np.float32))

for patient in range(len(data)):
    print(patient)
    for i in range(3): #fa, gm, rs
        
        graf_original_comp = len(list(nx.connected_components(nx.from_numpy_array(data[patient][:,:,i]))))
        print(f'graf{patient}', graf_original_comp)
        """
        if i == 2: #fem sparcification de la RS pq no té cap mena de sentit fer link salience per un graf amb diametre 1
            sw = data[patient][:,:,i]
            thr = sparsify_percentile(sw, percentile=percentil_RS)
            sw = np.where(sw >= thr, sw, 0) # ens quedem amb els valors majors al threshold
            sw = salience_weight(sw, data_type=3)
            # print('a', sw.size)
        """

        if i == 2: #fem sparcification de la RS pq no té cap mena de sentit fer link salience per un graf amb diametre 1
            sw = data[patient][:,:,i]
            sw = nx.from_numpy_array(sw)
            thr = thresholds_utils.graph_sparsification(sw, fraction=percentil_RS)
            thr = nx.to_numpy_array(thr)
            sw = salience_weight(thr, data_type=3)

            # print('a', sw.size)
        else: sw = salience_weight(data[patient], data_type=i) #aqui es pot posar data_type=0,1,2 segons si volem mirar FA GM o RS, predeterminat és 0
        
        g_prima_W = nx.from_numpy_array(sw)


        # kruskal_fals = add_edges_until_connected(g_prima_W) -----> no torna valors esperats encara, els weights creats per salience son extremadament iguals, no tenen sentit. especialment a RS
        g_prima_W = add_edges_until_connected(g_prima_W, comp = graf_original_comp)

        # original_edges(kruskal_fals, data, patient, i)
        original_edges(g_prima_W, data, patient, i)

        adj = nx.to_numpy_array(g_prima_W)
        np.save(f'./graphs/{datatype[i]}_{patient:04}_salience_kr_{target[patient]}_kruskalfals.npy', adj.astype(np.float32))

        """
        filename = f'./graphs/{datatype[i]}_{patient:04}_salience_krfals_{target[patient]}.graphml'
        nx.write_graphml(kruskal_fals, filename)

        filename = f'./graphs/{datatype[i]}_{patient:04}_salience_kr_{target[patient]}.graphml'
        nx.write_graphml(g_prima_W, filename)

        df = pd.DataFrame(adj)
        df.to_csv(f'./graphs/{datatype[i]}_{patient:04}_salience_kr_{target[patient]}.csv', index=False)
        """

        #print(filename)
        """
        for j in thresholds:
            g_prima = nx.from_numpy_array(sw)
            #print(g_prima.size())
            print('threshold:', j)
            print(len(np.unique(sw)), np.unique(sw))
            
            g_prima = delete_edges_numperc(g_prima, type_delete='per', percent=j)
            print(len(np.unique(sw)), np.unique(sw),'\n')
            original_edges(g_prima, data, patient, i)

            adj_x = nx.to_numpy_array(g_prima)
            np.save(f'./graphs/{datatype[i]}_{patient:04}_salience_threshold_{target[patient]}.npy', adj_x)
        """


t1 = time.time()
print(t1-t0,'segons')

