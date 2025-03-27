# Import libraries

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
import random
from collections import defaultdict

import utils_preproc


# Apply a threshold to a graph
def threshold_edge_weight(G, threshold):
    G_thresholded = G.copy()
    G_thresholded.remove_edges_from(nx.selfloop_edges(G_thresholded))
    edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if abs(w) < threshold]
    G_thresholded.remove_edges_from(edges_to_remove)
    return G_thresholded


# Calculate average density of graphs across a span of thresholds
def threshold_avg_density_search(x, scan_type, data):

    y = []

    for threshold in x:
        #print("Current threshold = ", threshold)
        avg_density = 0

        for undirected_graph in range(data.shape[0]):
            G = utils_preproc.adjacency_matrix_to_graph(data[undirected_graph,:,:,scan_type])
            G = threshold_edge_weight(G, threshold)

            avg_density += utils_preproc.calculate_density_graph(G)
            
        y.append(avg_density/data.shape[0])

    return y


# Function to calculate thresholded graphs from a set of adjacency matrices for multiple thresholds
def calculate_thresholded_graphs(adj_matrix_set, thresholds, method):
    """
    Calculate thresholded graphs from a set of adjacency matrices for multiple thresholds.

    Parameters:
    - adj_matrix_set (list): A list of adjacency matrices.
    - thresholds (list): A list of thresholds to apply.

    Returns:
    - thresholded_graphs (dict): A dictionary where keys are thresholds and values are lists of corresponding thresholded graphs.
    """
    thresholded_graphs = {}  # Initialize an empty dictionary to store results

    for threshold in thresholds:
        thr_graphs = []  # List to store thresholded graphs for the current threshold
        
        for adj_mat in adj_matrix_set:
            if len(adj_mat) < 76:
                print(len(adj_mat))
            G = utils_preproc.adjacency_matrix_to_graph(adj_mat)  # Convert adjacency matrix to graph
            if nx.number_of_nodes(G) < 76:
                print(f'number of nodes of original graph: {nx.number_of_nodes(G)}')
            filtered_graph = threshold_edge_weight(G, threshold)  # Apply thresholding
            thr_graphs.append(filtered_graph)  # Append the thresholded graph to the list
            if nx.number_of_nodes(filtered_graph) < 76:
                print(f'number of nodes of thresholded graph: {nx.number_of_nodes(filtered_graph)}')

        # Store results in corresponding lists for each threshold
        thresholded_graphs[method + "_" + str(threshold)] = thr_graphs

    return thresholded_graphs


