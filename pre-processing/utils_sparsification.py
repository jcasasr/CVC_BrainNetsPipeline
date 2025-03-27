# Import libraries

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import random
from collections import defaultdict
import utils_preproc


# Function to calculate the degree-preserving sparsificated graph
def degree_preserving_sparsification(graph, retain_fraction):
    sparsified_graph = nx.Graph()
    sparsified_graph.add_nodes_from(graph.nodes())
    
    for node in graph.nodes():
        # Get all edges incident to the node
        edges = list(graph.edges(node))
        num_edges_to_keep = int(len(edges) * retain_fraction)
        
        # Sample a subset of edges to keep
        sampled_edges = random.sample(edges, num_edges_to_keep)
        
        for u, v in sampled_edges:
            weight = graph[u][v]['weight'] if 'weight' in graph[u][v] else 1
            sparsified_graph.add_edge(u, v, weight=weight)
    
    return sparsified_graph


# Function to calculate thresholded graphs from a set of adjacency matrices for multiple thresholds
def calculate_sparsified_graphs(adj_matrix_set, fractions, method):
    """
    Calculate thresholded graphs from a set of adjacency matrices for multiple thresholds.

    Parameters:
    - adj_matrix_set (list): A list of adjacency matrices.
    - fractions (list): A list of retain fractions to apply.

    Returns:
    - sparsified_graphs (dict): A dictionary where keys are thresholds and values are lists of corresponding sparsified graphs.
    """
    sparsified_graphs = {}  # Initialize an empty dictionary to store results

    for fraction in fractions:
        spar_graphs = []  # List to store sparsified graphs for the current retain fraction
        
        for adj_mat in adj_matrix_set:
            if len(adj_mat) < 76:
                print(len(adj_mat))
            G = utils_preproc.adjacency_matrix_to_graph(adj_mat)  # Convert adjacency matrix to graph
            if nx.number_of_nodes(G) < 76:
                print(f'number of nodes of original graph: {nx.number_of_nodes(G)}')
            filtered_graph = degree_preserving_sparsification(G, fraction)  # Apply sparsification
            spar_graphs.append(filtered_graph)  # Append the sparsified graph to the list
            if nx.number_of_nodes(filtered_graph) < 76:
                print(f'number of nodes of sparsified graph: {nx.number_of_nodes(filtered_graph)}')

        # Store results in corresponding lists for each threshold
        sparsified_graphs[method + "_" + str(fraction)] = spar_graphs

    return sparsified_graphs
