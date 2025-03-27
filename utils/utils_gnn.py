import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def array_to_graph(data, node_embeddings, y):
    """
    Convert a dense NumPy adjacency matrix into a PyTorch Geometric Data object,
    incorporating node embeddings.

    Args:
    - data: NumPy array of shape (num_nodes, num_nodes), adjacency matrix.
    - node_embeddings: NumPy array of shape (num_nodes, num_features), node embeddings.
    - y: Target label (scalar or one-hot).

    Returns:
    - PyTorch Geometric Data object.
    """
    # Convert the dense adjacency matrix to a PyTorch tensor
    adj_tensor = torch.tensor(data, dtype=torch.float)
    
    # Convert dense tensor to edge_index and edge_weight
    edge_index, edge_weight = dense_to_sparse(adj_tensor)
    
    # Convert node embeddings to a PyTorch tensor
    x = torch.tensor(node_embeddings, dtype=torch.float)
    
    # Create the target label tensor
    y = torch.tensor([int(y)], dtype=torch.long)  # Classification target
    
    # Construct the PyTorch Geometric Data object
    graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    
    return graph_data

