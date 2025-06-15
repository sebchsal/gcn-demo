import torch
import networkx as nx
#Funcion original del codigo base 
def create_karate_dataset():
    """Carga el grafo del Club de Karate de Zachary y prepara tensores para PyTorch Geometric.

    Retorna:
        - x (torch.Tensor): Matriz de características [num_nodos, 1] con el grado de cada nodo.
        - edge_index (torch.Tensor): Tensor de aristas [2, num_aristas*2].
        - y (torch.Tensor): Etiquetas de nodos (0 o 1).
        - train_mask (torch.BoolTensor): Máscara booleana para nodos de entrenamiento.
        - test_mask (torch.BoolTensor): Máscara booleana para nodos de prueba.
        - G (networkx.Graph): Objeto grafo original.
    """
    G = nx.karate_club_graph() # Cargar el grafo
    # Crear índice de aristas bidireccional
    edge_index = torch.tensor(list(G.edges)).t().contiguous() 
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    # Características: grado de cada nodo
    degrees = dict(G.degree())
    x = torch.tensor([[degrees[i]] for i in range(len(G))], dtype=torch.float)
    # Etiquetas: 1 si el nodo pertenece al club 'Officer'
    y = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(len(G))], dtype=torch.long)
    # Definir división train/test
    train_mask = torch.zeros(len(G), dtype=torch.bool)
    train_nodes = [0, 33, 1, 2, 3, 8, 13, 19, 20, 22]
    train_mask[train_nodes] = True
    test_mask = ~train_mask

    return x, edge_index, y, train_mask, test_mask, G