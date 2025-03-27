# Import libraries

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
import random
from collections import defaultdict


# Function to write information into a pickle file
def save_graphs(file_path, information):
    with open(file_path, 'wb') as f:
        pickle.dump(information, f)


# Create a graph from the adjacency matrix and remove self-loops
def adjacency_matrix_to_graph(adj_matrix):

    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    # Apply absolute value transformation to all edge weights
    nx.set_edge_attributes(G, {e: abs(data['weight']) for e, data in G.edges.items()}, 'weight')

    '''
    G = nx.Graph()  # Use nx.DiGraph() for directed graphs
    
    # Add edges between nodes based on the adjacency matrix
    num_nodes = adj_matrix.shape[0]
    print('adj mat to graph; matrix number of nodes: ', num_nodes)
    for i in range(num_nodes):
        for j in range(i, num_nodes):  # (i, j) to avoid duplicates
            if adj_matrix[i][j] != 0:  # Non-zero means an edge exists
                G.add_edge(i, j, weight=adj_matrix[i][j])  # Add the edge with weight if applicable
    print('adj mat to graph; graph number of nodes: ', G.number_of_nodes())
    '''
    return G


# funció que agafa un graf i una pos, i dibuixa amb les arestes ja pintades i amb alfes
def edges_pintats(g, pos=None, color='plasma', tipus=''):

    if pos == None:
        pos = nx.spring_layout(g, k=2, iterations=50)
    
    fig, ax = plt.subplots(figsize=(12, 8))

    #dibuixar nodes
    nx.draw(g, pos, with_labels=True, node_color='black', font_color='white', node_size=100, font_size=10, ax=ax)

    #pesos de les arestes
    edge_weights = nx.get_edge_attributes(g, 'weight')

    weights = np.array(list(edge_weights.values()))  # agafem tots els pesos
    norm_weights = 1 - (0.9 * (weights.max() - weights) / (weights.max() - weights.min()))  # normalitzem de 0.1 a 1 per determinar alpha, que no volem que desapareguin

    #color de les arestes
    cmap = cm.get_cmap(color)

    # assignem colors i opacitat
    edges = g.edges()
    edge_colors = [cmap(norm_weights[i]) for i in range(len(edges))]
    edge_alphas = norm_weights

    #dibuixem arestes
    for i, (u, v) in enumerate(edges):
        nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], edge_color=[edge_colors[i]], width=1.5, alpha=edge_alphas[i], ax=ax)

    # Afegir una escala de colors (colorbar)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=weights.min(), vmax=weights.max()))

    cbar = plt.colorbar(sm, ax=ax)  # Afegir el colorbar a l'eix
    cbar.set_label('edge weight color grading')
    
    if type(tipus) == int:
        plots = ['FA', 'GM', 'RS']
        plt.title(f"Graf {plots[tipus]}:")
    else:
        plt.title(f"Graf {tipus}:")

    plt.show()

    #analisi de dades
    if nx.is_connected(g):
        print("Diàmetre:\t",nx.diameter(g))
    else:
        print("Hi ha components no connexes. Els diàmetres son:", [nx.diameter(g.subgraph(component)) for component in list(nx.connected_components(g))])
    print("Densitat:\t",nx.density(g))
    print("És arbre:\t",nx.is_tree(g))


# Calculate the density of a graph without self-loops
def calculate_density_graph(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.density(G)


# ---- graph measures ----

# Function to calculate the number of nodes and edges of a set of graphs
def get_number_of_nodes_and_edges(adj_matrices):
    """
    This function takes a list of adjacency matrices, converts them to graph objects, 
    and returns the number of nodes and edges for each graph.
    
    :param adj_matrices: A list of adjacency matrices (each as a 2D numpy array).
    :return: A list of tuples where each tuple contains (number_of_nodes, number_of_edges) for each graph.
    """
    graph_info = []

    # Iterate through each adjacency matrix
    for adj_matrix in adj_matrices:
        # Convert the adjacency matrix to a NetworkX graph
        G = nx.from_numpy_array(np.array(adj_matrix))
        G.remove_edges_from(nx.selfloop_edges(G))

        # Get the number of nodes and edges
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Append the info as a tuple (num_nodes, num_edges)
        graph_info.append((num_nodes, num_edges))

    return graph_info


# Function to calculate various graph measures
def calculate_graph_measures(G, original_num_nodes, original_num_edges):
    measures = {}
    measures['node_retention_percentage'] = (G.number_of_nodes() / original_num_nodes) * 100
    measures['edge_retention_percentage'] = (G.number_of_edges() / original_num_edges) * 100
    measures['is_connected'] = nx.is_connected(G)
    measures['average_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
    measures['average_clustering'] = nx.average_clustering(G)
    
    if G.number_of_nodes() > 1:  # To avoid division by zero for a graph with 1 or no nodes
        measures['density'] = calculate_density_graph(G)
    else:
        measures['density'] = 0  # Set density to 0 for trivial graphs

    if G.number_of_nodes() > 1 and measures['is_connected']:
        measures['diameter'] = nx.diameter(G)
        measures['average_shortest_path'] = nx.average_shortest_path_length(G)
    else:
        #measures['diameter'] = None
        #measures['average_shortest_path'] = None

        # Find all connected components
        connected_components = list(nx.connected_components(G))
    
        # Find the largest connected component
        largest_cc = max(connected_components, key=len)
    
        # Create a subgraph from the largest connected component
        G_largest_cc = G.subgraph(largest_cc).copy()
    
        # Now calculate the diameter and average shortest path length for the largest component
        measures['diameter'] = nx.diameter(G_largest_cc)
        measures['average_shortest_path'] = nx.average_shortest_path_length(G_largest_cc)
    
    return measures


# Function to calculate average graph measures (over relative number) from a list of thresholded graphs
def calculate_average_graph_measures(thresholded_graphs, original_graph_info):
    """
    Calculate average measures over a set of thresholded graphs.

    Parameters:
    - thresholded_graphs (list): A list of thresholded graphs.

    Returns:
    - avg_measures (dict): A dictionary containing average measures.
    """
    avg_measures = {
        'node_retention_percentage': 0,
        'edge_retention_percentage': 0,
        'is_connected': 0,
        'average_degree': 0,
        'average_clustering': 0,
        'density': 0,
        'diameter': 0,
        'average_shortest_path': 0,
    }

    # Counters for valid counts
    valid_counts = {
        'node_retention_percentage': 0,
        'edge_retention_percentage': 0,
        'is_connected': 0,
        'average_degree': 0,
        'average_clustering': 0,
        'density': 0,
        'diameter': 0,
        'average_shortest_path': 0,
    }

    # Counter for retrieving the original graph info
    info = 0

    # Loop over all thresholded graphs to calculate measures
    for filtered_graph in thresholded_graphs:
        original_num_nodes = original_graph_info[info][0]
        original_num_edges = original_graph_info[info][1]

        # Calculate measures for this graph
        measures = calculate_graph_measures(filtered_graph, original_num_nodes, original_num_edges)

        # Accumulate the values and increment valid counts
        avg_measures['node_retention_percentage'] += measures['node_retention_percentage']
        valid_counts['node_retention_percentage'] += 1

        avg_measures['edge_retention_percentage'] += measures['edge_retention_percentage']
        valid_counts['edge_retention_percentage'] += 1

        avg_measures['is_connected'] += 1 if measures['is_connected'] else 0  # Treat connectivity as 1/0
        valid_counts['is_connected'] += 1  # Every graph has an is_connected value

        avg_measures['average_degree'] += measures['average_degree']
        valid_counts['average_degree'] += 1

        avg_measures['average_clustering'] += measures['average_clustering']
        valid_counts['average_clustering'] += 1

        avg_measures['density'] += measures['density']
        valid_counts['density'] += 1
        
        if measures['diameter'] is not None:
            avg_measures['diameter'] += measures['diameter']
            valid_counts['diameter'] += 1

        if measures['average_shortest_path'] is not None:
            avg_measures['average_shortest_path'] += measures['average_shortest_path']
            valid_counts['average_shortest_path'] += 1
        
        info += 1

    # Compute the averages by dividing by the count of valid measures for each metric
    for measure in avg_measures.keys():
        if valid_counts[measure] > 0:
            avg_measures[measure] /= valid_counts[measure]
        else:
            avg_measures[measure] = None  # Set to None if no valid values were found for the measure

    return avg_measures


'''
# Function to calculate average (over the total number) of each measure across a set of graphs
def threshold_calculate_average_measures_over_full_graph(adj_matrix_set, threshold):
    avg_measures = {
        'node_retention_percentage': 0,
        'edge_retention_percentage': 0,
        'is_connected': 0,
        'average_degree': 0,
        'average_clustering': 0,
        'density': 0,
        'diameter': 0,
        'average_shortest_path': 0,
    }
    total_graphs = len(adj_matrix_set)

    for adj_mat in adj_matrix_set:
        G = utils_study.adjacency_matrix_to_graph(adj_mat)
        original_num_nodes = G.number_of_nodes()
        original_num_edges = G.number_of_edges()

        # Apply thresholding to the graph
        filtered_graph = threshold_edge_weight(G, threshold)

        # Calculate measures for this graph
        measures = utils_study.calculate_graph_measures(filtered_graph, original_num_nodes, original_num_edges)

        # Accumulate the values to compute the average
        avg_measures['node_retention_percentage'] += measures['node_retention_percentage']
        avg_measures['edge_retention_percentage'] += measures['edge_retention_percentage']
        avg_measures['is_connected'] += 1 if measures['is_connected'] else 0  # Treat connectivity as 1/0
        avg_measures['average_degree'] += measures['average_degree']
        avg_measures['average_clustering'] += measures['average_clustering']
        avg_measures['density'] += measures['density']
        
        if measures['diameter'] is not None:
            avg_measures['diameter'] += measures['diameter']
        if measures['average_shortest_path'] is not None:
            avg_measures['average_shortest_path'] += measures['average_shortest_path']

    # Compute the averages by dividing by the number of graphs
    avg_measures['node_retention_percentage'] /= total_graphs
    avg_measures['edge_retention_percentage'] /= total_graphs
    avg_measures['is_connected'] /= total_graphs  # Average connectivity as a percentage of connected graphs
    avg_measures['average_degree'] /= total_graphs
    avg_measures['average_clustering'] /= total_graphs
    avg_measures['density'] /= total_graphs
    
    if avg_measures['diameter'] > 0:
        avg_measures['diameter'] /= total_graphs
    else:
        avg_measures['diameter'] = None

    if avg_measures['average_shortest_path'] > 0:
        avg_measures['average_shortest_path'] /= total_graphs
    else:
        avg_measures['average_shortest_path'] = None

    return avg_measures
'''


# Function to plot and return the results for each measure based on thresholded graphs
def plot_graph_metrics(thresholded_graphs, original_graph_info, save_path, scan_type):
    """
    Plot the results for each graph metric based on thresholded graphs for different thresholds.

    Parameters:
    - thresholded_graphs (dict): A dictionary where keys are thresholds and values are lists of thresholded graphs.
    - save_path (str): The directory where the plots will be saved.

    Returns:
    - results (dict): A dictionary containing the results for each measure.
    """
    # Initialize results dictionary to store measure results for each threshold
    results = {
        'node_retention_percentage': [],
        'edge_retention_percentage': [],
        'is_connected': [],
        'average_degree': [],
        'average_clustering': [],
        'density': [],
        'diameter': [],
        'average_shortest_path': [],
    }

    # Loop over each threshold and calculate the average measures
    thresholds = list(thresholded_graphs.keys())  # Get thresholds from the dictionary
    for threshold in thresholds:
        #print(f"Current threshold: {threshold}")
        thr_graphs = thresholded_graphs[threshold]  # Get the thresholded graphs for the current threshold

        # Calculate average measures for the current set of thresholded graphs
        avg_measures = calculate_average_graph_measures(thr_graphs, original_graph_info)
        
        # Store results in corresponding lists for each measure
        for measure in results.keys():
            results[measure].append(avg_measures[measure])

    # Plot 1: Node Retention, Edge Retention, and Average Degree
    plt.figure(figsize=(10, 8))

    # List of different colors and line styles for the first plot
    colors1 = ['b', 'g', 'r']
    line_styles1 = ['-', '--', '-.']
    markers1 = ['o', 's', 'D']

    measures_plot1 = ['node_retention_percentage', 'edge_retention_percentage', 'average_degree']

    # Plot each measure for the first plot
    for i, measure in enumerate(measures_plot1):
        plt.plot(thresholds, results[measure], marker=markers1[i], linestyle=line_styles1[i], color=colors1[i], label=measure.replace('_', ' ').title())

    # Set plot title and labels
    plt.title(scan_type + " - Node Retention, Edge Retention, and Average Degree vs. Threshold", fontsize=14)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Measure Value", fontsize=12)

    # Enable grid and legend
    plt.grid(True)
    plt.legend(loc="best", fontsize=10)
    
    # Show the first plot
    plt.savefig(save_path + '1.png')
    #plt.show()

    # Plot 2: Other measures (Clustering, Density, Diameter, Average Shortest Path)
    plt.figure(figsize=(10, 8))

    # List of different colors and line styles for the second plot
    colors2 = ['c', 'm', 'y', 'k']
    line_styles2 = ['-', '--', '-.', ':']
    markers2 = ['^', 'v', '<', '>']

    measures_plot2 = ['average_clustering', 'density', 'diameter', 'average_shortest_path']

    # Plot each measure for the second plot
    for i, measure in enumerate(measures_plot2):
        plt.plot(thresholds, results[measure], marker=markers2[i], linestyle=line_styles2[i], color=colors2[i], label=measure.replace('_', ' ').title())

    # Set plot title and labels
    plt.title(scan_type + " - Clustering, Density, Diameter, and Shortest Path vs. Threshold", fontsize=14)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Measure Value", fontsize=12)

    # Enable grid and legend
    plt.grid(True)
    plt.legend(loc="best", fontsize=10)

    # Show the second plot
    plt.savefig(save_path + '2.png')
    #plt.show()

    return results


# ---- node measures ----

# Function to create a new dict including the un-thresholded gaphs
def append_original_graphs_to_threshold(thresholded_graphs, original_graphs):
    """
    Append the original graphs to the entry for threshold 0 in the given dictionary.

    Parameters:
    - thresholded_graphs (dict): A dictionary where keys are thresholds and values are lists of graphs.
    - original_graphs (list): A list of adjacency matrices of original graphs to append.

    Returns:
    - dict: Updated dictionary with original graphs appended to threshold 0, sorted by keys.
    """
    # Create a copy to avoid mutating the original dictionary
    thresholded_graphs = thresholded_graphs.copy()
    
    # Check if threshold 0 exists in the dictionary
    if 'original' not in thresholded_graphs:
        # If not, initialize it with an empty list
        thresholded_graphs['original'] = []

        graphs = []
        for adj_mat in original_graphs:
            G = adjacency_matrix_to_graph(adj_mat)  # Convert adjacency matrix to graph
            graphs.append(G)  # Append the thresholded graph to the list

        # Append the original graphs to 'original
        thresholded_graphs['original'].extend(graphs)

        return thresholded_graphs
        #return {key: thresholded_graphs[key] for key in sorted(thresholded_graphs)}
    else:
        print("Original graphs were already in the list.")
        return thresholded_graphs


# Function to calculate node measures from one graph
def calculate_node_measures(G, method):

    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    #eigenvector_centrality = nx.eigenvector_centrality(G)

    '''
    if scan_type == 'RS':

        # Store centrality measures in a dictionary for easy access
        centralities = {
            'Degree_Centrality': degree_centrality,
            'Betweenness_Centrality': betweenness_centrality,
            'Closeness_Centrality': closeness_centrality,
            'Eigenvector_Centrality': eigenvector_centrality
        }
    else:
        pagerank_centrality = nx.pagerank(G)
    '''
    if method == 'sal':
#        eigenvector_centrality = nx.hits(G)
        
        # Store centrality measures in a dictionary for easy access
        centralities = {
            'Degree_Centrality': degree_centrality,
            'Betweenness_Centrality': betweenness_centrality,
            'Closeness_Centrality': closeness_centrality,
     #       'Eigenvector_Centrality': eigenvector_centrality,
            #'PageRank': pagerank_centrality, clustering
        }
    else:
        eigenvector_centrality = nx.eigenvector_centrality(G)

        # Store centrality measures in a dictionary for easy access
        centralities = {
            'Degree_Centrality': degree_centrality,
            'Betweenness_Centrality': betweenness_centrality,
            'Closeness_Centrality': closeness_centrality,
            'Eigenvector_Centrality': eigenvector_centrality,
            #'PageRank': pagerank_centrality
        }

    return centralities


# Helper function to calculate and sort measures for a single graph
def calculate_sorted_measures(graph, method):
    # Calculate measures
    measures = calculate_node_measures(graph, method)

    # Sort each measure by node name
    sorted_measures = {measure_name: dict(sorted(measure_values.items()))
                       for measure_name, measure_values in measures.items()}
    
    return sorted_measures

# Main function to calculate node measures per graph across a set of graphs
def calculate_node_measures_per_graph(thresholded_graphs, method):
    """
    Calculates and sorts node measures for each graph per threshold.
    
    Parameters:
    - thresholded_graphs (dict): Dictionary with thresholds as keys and list of graphs as values.
    - scan_type: Type of scan or measure to calculate.

    Returns:
    - dict: A dictionary where each threshold maps to a list of sorted measures per graph.
    """
    # For each threshold, apply calculate_sorted_measures to each graph and collect results
    return {
        threshold: [calculate_sorted_measures(graph, method) for graph in graphs]
        for threshold, graphs in thresholded_graphs.items()
    }


# Function to calculate node measures from a set of graphs; triple for to track threshold
#  (equivalent to calculate_node_measures_per_graph)
def calculate_node_measures_per_graph_triple_for(thresholded_graphs, scan_type):
    # Initialize a dictionary to store the node measures for each graph per threshold
    all_measures = {}

    # Iterate over each threshold and its corresponding graphs
    for threshold, graphs in thresholded_graphs.items():
        measures_for_threshold = []
        print(threshold)

        # Iterate through each graph in the current threshold
        for filtered_graph in graphs:
            # Calculate measures for this graph
            measures = calculate_node_measures(filtered_graph, scan_type)

            # Sort the node measures by node name
            for measure_name, measure_values in measures.items():
                # Sort the dictionary based on the node name (keys)
                sorted_measures = dict(sorted(measure_values.items()))
                measures[measure_name] = sorted_measures

            # Append the measures for this graph to the list for the current threshold
            measures_for_threshold.append(measures)

        # Store the measures for all graphs of this threshold
        all_measures[threshold] = measures_for_threshold

    return all_measures


# Function to calculate the average node measures for each threshold across all graphs
def calculate_average_node_measures_per_threshold(all_measures):
    """
    Calculate the average node measures for each threshold across all graphs,
    and store them in the format: threshold -> measure -> node -> value.

    Parameters:
    - all_measures (dict): A nested dictionary where keys are thresholds and values are lists
                           of dictionaries containing measures for each graph.

    Returns:
    - avg_measures (dict): A nested dictionary where keys are thresholds, and each threshold 
                           maps to another dictionary with metrics as keys, which then map to 
                           dictionaries of nodes and their average measures.
    """
    avg_measures = {}

    # Iterate through each threshold and its corresponding list of graph measures
    for threshold, measures_list in all_measures.items():
        # Initialize a dictionary to store sums and counts for each measure
        sum_measures = {}
        count_measures = {}

        # Iterate through each graph's measures in the current threshold
        for measures in measures_list:
            for measure_name, node_values in measures.items():
                # Initialize measure in the sum and count dictionaries if not present
                if measure_name not in sum_measures:
                    sum_measures[measure_name] = {}
                    count_measures[measure_name] = {}

                # Iterate through each node in the measure
                for node, value in node_values.items():
                    # Initialize the node in the measure's sum and count dictionaries if not present
                    if node not in sum_measures[measure_name]:
                        sum_measures[measure_name][node] = 0
                        count_measures[measure_name][node] = 0

                    # Sum the values and count occurrences for averaging
                    sum_measures[measure_name][node] += value
                    count_measures[measure_name][node] += 1

        # Compute the averages for each measure
        avg_measures[threshold] = {}
        for measure_name, nodes in sum_measures.items():
            avg_measures[threshold][measure_name] = {}
            for node, total_value in nodes.items():
                # Calculate the average
                avg_measures[threshold][measure_name][node] = total_value / count_measures[measure_name][node]

    return avg_measures


# Function to find which nodes appear in the top N for all centrality metrics
def find_common_top_n_nodes(centralities, N):
    # Dictionary to store the top N nodes for each measure
    top_n_nodes_per_measure = {}

    # Extract the top N nodes for each centrality measure
    for measure, values in centralities.items():
        # Sort the nodes by centrality value (descending) and select the top N
        top_n_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)[:N]
        # Store only the node indices (not the values)
        top_n_nodes_per_measure[measure] = set(node for node, _ in top_n_nodes)

    # Find common nodes across all centrality measures
    # Initialize the common nodes set with the first measure's top N nodes
    common_nodes = set.intersection(*top_n_nodes_per_measure.values())

    return common_nodes


# Function to find top nodes across various metrics
def find_common_nodes_across_metrics(node_measures, N):
    common_nodes = {}
    for thr in node_measures.keys():
        common_nodes[thr] = find_common_top_n_nodes(node_measures[thr], N)
    return common_nodes


# Function to plot the centrality metrics for all the nodes across multiple graph sets, highlighting common nodes.
def plot_all_nodes_across_sets_with_commons(all_nodes, common_nodes, average_centralities_per_set, metric_name, save_path, scan_type):
    """
    - all_nodes (set): Set of all nodes to be plotted.
    - common_nodes (set): Set of nodes that are common across all graph sets and should be highlighted.
    - average_centralities_per_set (dict): Dictionary where the keys are graph set names (e.g., different thresholds),
                                           and the values are dictionaries containing the average centrality values 
                                           for each node in that graph set.
    - metric_name (str): The name of the centrality metric being plotted (e.g., 'Degree Centrality', 'Closeness Centrality').
    """
    # Sort the common nodes
    sorted_all_nodes = sorted(all_nodes)
    sorted_common_nodes = sorted(common_nodes)
    
    # Prepare a figure for plotting
    plt.figure(figsize=(10, 5))

    # Extract graph set names
    graph_sets = list(average_centralities_per_set.keys())

    # Create a color map for each graph set to mantain same colours
    color_map = {}
    for i, graph_set in enumerate(graph_sets):
        color_map[graph_set] = plt.get_cmap('tab10')(i)  # Assign a color from a colormap (e.g., tab10)
    
    # For each graph set, plot the average centrality values of the common nodes
    for graph_set in graph_sets:
        centralities = average_centralities_per_set[graph_set]

        values = [centralities.get(node, 0) for node in sorted_all_nodes]  # Get centrality value or 0 if node is missing
        plt.plot(sorted_all_nodes, values, label=f'{graph_set}', marker='.', alpha=0.8, color=color_map[graph_set])

        values_specials = [centralities.get(node, 0) for node in sorted_common_nodes]  # Get centrality value or 0 if node is missing
        plt.plot(sorted_common_nodes, values_specials, marker='*', markersize=10, linestyle='None', color=color_map[graph_set])

    # Labeling the plot
    plt.title(scan_type + ' - ' + metric_name)
    plt.xlabel('Nodes')
    plt.ylabel('Average Centrality Value')
    plt.xticks(sorted_all_nodes, rotation=45)  # Set x-tick labels to sorted nodes directly
    plt.legend()  # Add a legend to differentiate graph sets
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(save_path)
    #plt.show()


# Function to automatically call the 'plot_all_nodes_across_sets_with_commons' for all metrics
# checking for empty metrics (bc eigenvector centrality is not computed for saliency graphs)
def plot_node_metrics_per_threshold(all_nodes, all_common_nodes, avg_node_measures, save_path, scan_type):

    one_key = list(avg_node_measures.keys())[0]
    metrics_list = list(avg_node_measures[one_key].keys())

    for metric in metrics_list:
        average_centralities_per_set = {}
        
        for thr in avg_node_measures.keys():
            # Check if the metric is in the dictionary for this threshold
            if metric in avg_node_measures[thr]:
                average_centralities_per_set[thr] = avg_node_measures[thr][metric]

        # Only call the plotting function if there is data for the metric
        if average_centralities_per_set:
            plot_all_nodes_across_sets_with_commons(all_nodes, all_common_nodes, average_centralities_per_set, metric, save_path + metric, scan_type)


# Function to save a list of graphs into different npy arrays
def save_graphs_as_npy(graphs, path, prefix):
    """
    Saves a list of NetworkX graphs as .npy adjacency matrices.
    
    Parameters:
        graphs (list): A list of NetworkX graphs.
        prefix (str): Prefix for the saved .npy files.
        
    Returns:
        None
    """
    for i, graph in enumerate(graphs):
        # Convert graph to an adjacency matrix
        adj_matrix = nx.to_numpy_array(graph)
        
        # Save the adjacency matrix as a .npy file
        filename = f"{path}{prefix}_{i:04}.npy"
        np.save(filename, adj_matrix)
        #print(f"Graph {i} saved as {filename}")


# Function to save a list of graphs into a single npy array
def save_all_graphs_as_npy(graphs, save_path):
    """
    Saves a list of NetworkX graphs as a single .npy file containing all adjacency matrices.
    
    Parameters:
        graphs (list): A list of NetworkX graphs.
        save_path (str): The file path where the .npy file will be saved.
        
    Returns:
        None
    """
    # Convert each graph to an adjacency matrix
    adj_matrices = [nx.to_numpy_array(graph) for graph in graphs]
    
    # Stack all adjacency matrices along a new axis to create a 3D array
    adj_matrices_array = np.array(adj_matrices)
    
    # Save the 3D array as a single .npy file
    np.save(save_path, adj_matrices_array)
    print(f"Graphs saved in: {save_path}")

# Example usage:
# Assume we have a list of NetworkX graphs
# graphs = [nx.erdos_renyi_graph(10, 0.5), nx.complete_graph(5), nx.c
