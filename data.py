import torch
import networkx as nx

def create_karate_dataset():
    G = nx.karate_club_graph()
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    degrees = dict(G.degree())
    x = torch.tensor([[degrees[i]] for i in range(len(G))], dtype=torch.float)
    y = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(len(G))], dtype=torch.long)

    train_mask = torch.zeros(len(G), dtype=torch.bool)
    train_nodes = [0, 33, 1, 2, 3, 8, 13, 19, 20, 22]
    train_mask[train_nodes] = True
    test_mask = ~train_mask

    return x, edge_index, y, train_mask, test_mask, G