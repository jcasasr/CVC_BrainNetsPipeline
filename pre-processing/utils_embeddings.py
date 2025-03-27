import os
import csv
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import random

def merge_dicts(*dicts):
    # Get all unique keys across all dictionaries
    all_keys = set(key for d in dicts for key in d)
    
    # Create the merged dictionary with lists of values from each input dictionary
    merged_dict = {key: [d.get(key) for d in dicts] for key in all_keys}
    
    return merged_dict


def append_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].append(value)
        else:
            print("Error keys: dict2 key does not exist in dict1. Creating new key.")
            dict1[key] = [value]

    return(dict1)


def betweenness_centrality_inv(G, weight='weight'):
    # invertim pq les mes grans es tornin les mes petites
    for u, v, w in G.edges(data=True):
        w[weight] = -w[weight]
    
    return nx.betweenness_centrality(G)


def clustering_inv(G, weight='weight'):
    # invertim pq les mes grans es tornin les mes petites
    for u, v, w in G.edges(data=True):
        w[weight] = -w[weight]
    
    return nx.clustering(G)


# Cache to avoid recomputation
# @lru_cache(maxsize=None)
def calculate_metrics(G):
    """
    Calculates and returns all metrics for a given graph.
    This function is cached to prevent redundant calculations.
    """
    m1 = nx.degree_centrality(G)
    m2 = betweenness_centrality_inv(G, weight='weight')
    m3 = clustering_inv(G, weight='weight')
    m4 = all_random_walks(G, tfidf=False)
    # m4 = dict(enumerate(array, start=0))
    # print(m4)
    return merge_dicts(m1, m2, m3, m4)


def save_csv_file(metrics, output_dir, file_name):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    file_to_open = os.path.join(output_dir, file_name)
    with open(file_to_open, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Node', 'Degree Centrality', 'Betweenness Centrality', 'Clustering Coefficient', 'Random Walk'
        ])
        for key, values in metrics.items():
            writer.writerow([key] + values)


def generate_random_walks(graph, walks_per_node=3, walk_length=8, tfidf=False, return_len=False):
    """
    Genera random walks per a cada node del graf.
    """
    walks = []
    nodes = list(graph.nodes())

    for node in nodes:
        for _ in range(walks_per_node):
            walk = [node]  # Comença pel node inicial
            current = node
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(current))
                if not neighbors:  # Si no té veïns, acaba el camí
                    break
                next_node = random.choice(neighbors)
                walk.append(next_node)
                current = next_node
            walks.append(walk if not tfidf else " ".join(map(str, walk)))
    
    if return_len:
        return walks, len(walks)  # Retorna camins i nombre total
    return walks


def all_random_walks(G, tfidf=False):
    """
    Comptem la freqüència normalitzada de nodes visitats en random walks.
    """
    dict_counter = {}  # Inicialitzem un comptador
    total_visits = 0  # Total de visites als nodes
    
    # Generem random walks
    walks = generate_random_walks(G, tfidf=tfidf)
    # print('gerard', walks)
    # Iterem pels camins
    for walk in walks:
        # print('aaaaaaa', walk)
        if tfidf: walk = walk.split()
        
        for node in walk:
            if node not in dict_counter:
                dict_counter[int(node)] = 0
            dict_counter[int(node)] += 1
            total_visits += 1
            # print(total_visits)

    #apliquem normalitzacio per tenirho entre 0-1 del chill
    # print(total_visits)
    # print('abans', dict_counter)
    for node in dict_counter:
        dict_counter[node] /= total_visits
    # print('despressssssssssssssssssssssssssssssssssssssssssssssssssssssssssss', dict_counter)
    return dict_counter



def walks_tfidf(walks, num_nodes):
    # Inicialitza comptadors per a TF i DF
    tf = [Counter(walk.split()) for walk in walks]
    df = Counter()
    
    # Calcula el DF per a cada node (quants walks contenen cada node)
    for counter in tf:
        for node in counter.keys():
            df[node] += 1
    
    # Total de random walks
    total_walks = len(walks)
    
    # Array per emmagatzemar el TF-IDF final per cada node
    tfidf = np.zeros(num_nodes)
    
    # Calcula el TF-IDF per a cada node
    for node in range(num_nodes):
        node_str = str(node)
        # IDF: si el node no apareix en cap walk, el seu idf és 0
        idf = math.log(total_walks / (df[node_str] + 1)) if df[node_str] > 0 else 0
        # TF-IDF: suma de tf-idf per cada walk on apareix el node
        tfidf[node] = sum((counter[node_str] / sum(counter.values())) * idf 
                          for counter in tf if node_str in counter)
        
    # Normalització min-max
    min_tfidf, max_tfidf = tfidf.min(), tfidf.max()
    if max_tfidf > min_tfidf:  # Per evitar divisió per zero
        tfidf = (tfidf - min_tfidf) / (max_tfidf - min_tfidf)

    return tfidf

#comptem simplement la freqüència normalitzada de quant apareixen en els 
def frequency_counter(ll):
    elem = [element for subllista in ll for element in subllista]
    compt = Counter(elem)
    compt = dict(sorted(compt.items()))
    compt_list = np.array(list(compt.values())) # aqui passem a tenir una llista
    compt_list = compt_list / len(ll) # normalitzem dividint el nombre de cops que apareixen en total, per el nombre 

    return compt_list

