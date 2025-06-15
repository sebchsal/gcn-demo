import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#Funcion original del codigo base
class SimpleGCN(nn.Module):
    """Red convolucional de grafos sencilla de 2 capas.

    Args:
        - input_dim (int): Dimensión de las características de entrada.
        - hidden_dim (int): Número de unidades en la capa oculta.
        - output_dim (int): Número de clases de salida.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # Primera capa de convolución de grafos
        self.conv2 = GCNConv(hidden_dim, output_dim)  # Segunda capa de convolución de grafos

    def forward(self, x, edge_index):
        """Propagación hacia adelante en el GCN.

        Args:
            - x (torch.Tensor): Matriz de características de nodos.
            - edge_index (torch.Tensor): Índices de conectividad del grafo.

        Retorna:
            - logits (torch.Tensor): Puntuaciones crudas por clase y nodo.
            - embeddings (torch.Tensor): Representaciones ocultas de los nodos.
        """
        h1 = F.relu(self.conv1(x, edge_index))  # Aplicar primera convolución + ReLU
        h2 = self.conv2(h1, edge_index) # Aplicar segunda convolución para obtener logits finales
        return h2, h1