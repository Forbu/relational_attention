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
from torch_scatter import scatter

class MLP(nn.Module):

    def __init__(self, n_embd, dropout=0.):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

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

        q_ij = q_node[edges_index[0]] * q_edge  # shape (E, output_size)
        k_ij = k_node[edges_index[1]] * k_edge  # shape (E, output_size)
        v_ij = v_node[edges_index[1]] * v_edge  # shape (E, output_size)

        # now we compute the attention
        # we use the dot product between q_ij and k_ij.T
        # and then we apply softmax
        output_edges = F.scaled_dot_product_attention(
            q_ij.unsqueeze(0), k_ij.unsqueeze(0), v_ij.unsqueeze(0)
        )

        output_edges = output_edges.squeeze(0)

        # now we need to sum the edges values for each node
        output_nodes = scatter(
            src=output_edges, index=edges_index[1], dim=0, reduce="sum"
        )

        return output_nodes, output_edges
    
class MultiHeadRelationalAttention(nn.Module):
    """
    This is a module class for multi head relational attention
    """
    
    def __init__(self, nb_head=4, n_embd=384):
        self.nb_head = nb_head
        self.n_embd = n_embd
        
        # we define the layers
        self.WQ_node = nn.Linear(n_embd, n_embd)
        self.WQ_edge = nn.Linear(n_embd, n_embd)

        self.WK_node = nn.Linear(n_embd, n_embd)
        self.WK_edge = nn.Linear(n_embd, n_embd)

        self.WV_node = nn.Linear(n_embd, n_embd)
        self.WV_edge = nn.Linear(n_embd, n_embd)
    
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
        
        # now we create the edges values
        q_ij = q_node[edges_index[0]] * q_edge  # shape (E, output_size)
        k_ij = k_node[edges_index[1]] * k_edge  # shape (E, output_size)
        v_ij = v_node[edges_index[1]] * v_edge  # shape (E, output_size)
        
        # now we create the heads
        q_ij = q_ij.view(self.nb_head, 1, self.n_embd // self.nb_head)
        k_ij = k_ij.view(self.nb_head, 1, self.n_embd // self.nb_head)
        v_ij = v_ij.view(self.nb_head, 1, self.n_embd // self.nb_head)
        
        # now we compute the attention
        # we use the dot product between q_ij and k_ij.T
        # and then we apply softmax
        output_edges = F.scaled_dot_product_attention(
            q_ij, k_ij, v_ij
        )
        
        output_edges = output_edges.squeeze(1)
        
        # reshape the output edges
        output_edges = output_edges.view(-1, self.n_embd)
        
        # now we need to sum the edges values for each node
        output_nodes = scatter(
            src=output_edges, index=edges_index[1], dim=0, reduce="sum"
        )
        
        return output_nodes, output_edges