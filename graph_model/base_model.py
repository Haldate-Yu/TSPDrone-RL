import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge, GINConv


class Net_GCN(nn.Module):
    def __init__(self, input_dims, num_layers, hidden_dims, output_dims, dropout=0.1):
        super().__init__()

        self.input_dims = input_dims
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.dropout = dropout

        self.node_in = nn.Linear(input_dims, hidden_dims)
        self.node_out = nn.Linear(hidden_dims, output_dims)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dims, hidden_dims))

    def forward(self, batch_data):
        output_list = []

        for i in range(len(batch_data)):
            data = batch_data[i]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            x = self.node_in(x)

            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            output = self.node_out(x)
            output_list.append(output)
        return torch.stack(output_list, dim=0)
