import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool


class GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 64  

        # Primera i segona capa GCN
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim//2)

        # Capa lineal final (output de 4 classes)
        self.lin1 = torch.nn.Linear(hidden_dim//2, 4)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # Primera capa GCN
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        # Segona capa GCN
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Agregació global per grafs
        x = global_max_pool(x, batch)
        
        # Classificació final
        x = self.lin1(x)

        return x

# class GCN(torch.nn.Module): ###MODEL GAT, sense canvi de nom per evitar canviar noms de crides
#     def __init__(self, input_dim, heads=8):
#         super().__init__()
#         hidden_dim = 64  # Mateixa mida que en GCN
        
#         # Primera capa GAT amb caps d'atenció
#         self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=0.6)
        
#         # Capa lineal final (output de 2 classes)
#         self.lin1 = torch.nn.Linear(hidden_dim, 4)

#     def forward(self, data):
#         x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

#         # Passar per GATConv
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)  # S'acostuma a fer servir ELU en GAT

#         # Agregació global per grafs
#         x = global_max_pool(x, batch)
        
#         # Classificació final
#         x = self.lin1(x)

#         return x
    

# class GCN(torch.nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.conv1 = GCNConv(input_dim, 128)
#         self.lin1 = torch.nn.Linear(128, 64)
#         self.lin2 = torch.nn.Linear(64, 16)
#         self.lin3 = torch.nn.Linear(16, 2)

#     def forward(self, data):
#         x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
#         x = self.conv1(x, edge_index, edge_weight)
#         x = F.relu(x)
#         x = global_max_pool(x, batch)
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = self.lin2(x)
#         x = F.relu(x)
#         x = self.lin3(x)

#         return x
