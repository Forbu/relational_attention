"""
In this module, we define the layers used in the model, as define in https://arxiv.org/pdf/2210.05062.pdf

Basicly we have :
qij = [ni, eij ]WQ 
kij = [nj , eij ]WK 
vij =  = [nj , eij]WV 

and so at the end we have :
ai = softmax(qij * kij.T) * vij
"""

import torch
from torch import nn
from torch.nn import functional as F

class RelationalAttention(nn.Module):
    """
    Attention layer, as defined in https://arxiv.org/pdf/2210.05062.pdf
    """
    def __init__(self, node_size, edge_size, output_size):
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.output_size = output_size
        
        # now we project nodes and edges into a common space for 
        # the key, query and value
        self.WQ_node = nn.Linear(node_size, output_size)
        self.WQ_edge = nn.Linear(edge_size, output_size)
        
        self.WK_node = nn.Linear(node_size, output_size)
        self.WK_edge = nn.Linear(edge_size, output_size)
        
        self.WV_node = nn.Linear(node_size, output_size)
        self.WV_edge = nn.Linear(edge_size, output_size)
        
    def forward(self, nodes, edges_index, edges_values):
        """
        params:
            nodes: tensor of shape (N, node_size)
            edges_index: tensor of shape (2, E)
            edges_values: tensor of shape (E, edge_size)
        """
        # project nodes and edges into a common space for
        # the key, query and value
        q_node = self.WQ_node(nodes)
        k_node = self.WK_node(nodes)
        v_node = self.WV_node(nodes)
        
        # we need to get the edges values for each node
        # so we use the index to get the edges values
        # for each node
        q_edge = self.WQ_edge(edges_values)
        k_edge = self.WK_edge(edges_values)
        v_edge = self.WV_edge(edges_values)
        
        q_ij = q_node[edges_index[0]] * q_edge # shape (E, output_size)
        k_ij = k_node[edges_index[1]] * k_edge # shape (E, output_size)
        v_ij = v_node[edges_index[1]] * v_edge # shape (E, output_size)
        
        # now we compute the attention
        # we use the dot product between q_ij and k_ij.T
        # and then we apply softmax
        output_edges = F.scaled_dot_product_attention(q_ij, k_ij, v_ij)
        output_nodes = torch.scatter_add_(0, edges_index[0], output_edges)
        
        return output_nodes, output_edges
    




