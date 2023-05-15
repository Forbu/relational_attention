"""
test for layers module
"""

import torch
import pytest

from relational_attention.layers import RelationalAttention, MultiHeadRelationalAttention

# fix torch seed
torch.manual_seed(0)

# init fixtures 
@pytest.fixture
def init_graph():
    
    # here we do graph initialization
    nb_nodes = 10
    dim_nodes = 384
    
    nodes = torch.randn(nb_nodes, dim_nodes)
    
    # here we do edges initialization
    nb_edges = 20
    dim_edges = 384
    
    edges_index = torch.randint(0, nb_nodes, (2, nb_edges)).long()
    edges = torch.randn(nb_edges, dim_edges)
    
    return nodes, edges_index, edges

# not test per default
def test_relationallayer(init_graph):
    nodes, edges_index, edges = init_graph
    
    # here we do the forward pass
    layer = RelationalAttention(node_size=nodes.shape[1], edge_size=edges.shape[1], output_size=10)
    out_nodes, out_edges = layer(nodes, edges_index, edges)
    
    # we check the output shape
    assert out_nodes.shape == (nodes.size(0), 10)
    assert out_edges.shape == (edges.size(0), 10)
    
    
def test_multiheadrelationalattention(init_graph):
    nodes, edges_index, edges = init_graph
    
    # here we do the forward pass
    layer = MultiHeadRelationalAttention(nb_head=4, n_embd=384)
    
    out_nodes, out_edges = layer(nodes, edges_index, edges)
    
    # we check the output shape
    assert out_nodes.shape == (nodes.size(0), 384)
    assert out_edges.shape == (edges.size(0), 384)
        
