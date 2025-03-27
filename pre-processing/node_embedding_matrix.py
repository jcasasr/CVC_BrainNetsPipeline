"""
# Description:
# This script processes node embeddings and node volumes for multiple subjects, organizes the data, 
# and saves it in a single npy file.

# Functionality:
# 1. Reads embedding files from the 'node_embeddings' folder, where each file corresponds to a subject.
#    - Each file contains rows representing nodes, with the first column as the node ID and subsequent 
#      columns as embedding values.
# 2. Groups the embeddings by subject and node, storing them in a structured dictionary.
# 3. Loads node volume data from the 'node_volumes' folder and combines it with the corresponding 
#    node embeddings.
# 4. Flattens and concatenates embeddings and node volume data for each node.
# 5. Sorts the data by subject and saves it as a 3D NumPy array in the 'data' folder:
#    - Array shape: (num_subjects, num_nodes, total_features)
#      - num_subjects: Number of processed subjects.
#      - num_nodes: Fixed number of nodes (76).
#      - total_features: Length of concatenated embeddings and volume features.

# Input:
# - Embedding files in './node_embeddings/' folder (CSV files, one per subject).
# - Node volume files in './node_volumes/' folder (CSV files, one per subject).

# Output:
# - A single NumPy file ('node_embeddings.npy') saved in the './data/' folder.
"""
# Define imports
import numpy as np
import os
import pandas as pd

basepath = './'
basepath_data = os.path.join(basepath, 'data')
basepath_embeddings = os.path.join(basepath, 'node_embeddings')
basepath_node_vol = os.path.join(basepath, 'node_volumes')


# Generate the node embeddings matrix
num_nodes = 76

# Initialize a dictionary to store embeddings by file num
embed_mega_dict = {}

# Iterate through each file in the embeddings folder, SORTED
for file in sorted(os.listdir(basepath_embeddings)):
    # Extract the file identifier (num)
    num = int(file[:4])
    
    # Read the embeddings (df) and node volumes (df2)
    df = pd.read_csv(os.path.join(basepath_embeddings, file), header=0)  # assuming header is the first row
    
    # Initialize the list for the current `num` if it doesn't already exist
    if num not in embed_mega_dict:
        embed_mega_dict[num] = [[] for _ in range(num_nodes)]
    
    # Append embedding values (df) for each node
    for i in range(len(df)):
        node_id = int(df.iloc[i, 0])  # Node ID (row index in embedding array)
        embedding = df.iloc[i, 1:].values  # Select only the embedding values (excluding the first column)
        
        # Append embedding values for the node
        embed_mega_dict[num][node_id].append(embedding)

# Reshape and concatenate the embeddings for each node
for num in embed_mega_dict:
    # Load the node volumes for the current `num` (patient)
    df2 = pd.read_csv(os.path.join(basepath_node_vol, f"{num:04}_node_vol.csv"), header=None)

    for node_id in range(num_nodes):
        # Flatten the node embeddings from a list of arrays into a single 1D array
        node_embeddings = np.concatenate(embed_mega_dict[num][node_id], axis=0)

        # Get the node volume for the current node
        node_vol = df2.iloc[node_id].values  # Select the node volume values as a 1D array

        # Concatenate the node volume with the embeddings
        embed_mega_dict[num][node_id] = np.concatenate([node_embeddings, node_vol], axis=0)

# Sort the dictionary by `num` and convert it into a sorted list
embed_mega_array_sorted = [embed_mega_dict[num] for num in sorted(embed_mega_dict.keys())]

# Now `embed_mega_array_sorted` will have the combined data and the desired shape
print(np.array(embed_mega_array_sorted).shape)

# Save the embeddings to a .npy file
np.save(os.path.join(basepath_data, 'node_embeddings.npy'), embed_mega_array_sorted)
print("Embeddings saved to /data folder as `node_embeddings.npy`")
