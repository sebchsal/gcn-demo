import torch          
import torch.nn.functional as F
import pandas as pd
import networkx as nx
#Funcion original del codigo base mejorada
def analyze_node_importance(G, model, x, edge_index):
    """Analiza la importancia de cada nodo combinando las predicciones del modelo con métricas de centralidad del grafo.

    Parámetros:
        - G (networkx.Graph): El grafo de entrada.
        - model (torch.nn.Module): Modelo GCN entrenado.
        - x (torch.Tensor): Tensor de características de los nodos.
        - edge_index (torch.Tensor): Tensor de índices de aristas.

    Retorna:
    pd.DataFrame: DataFrame con una fila por nodo que contiene:
        - node: Identificador del nodo.
        - degree: Grado del nodo.
        - predicted_class: Clase predicha por el modelo (0 o 1).
        - confidence: Probabilidad asociada a la clase elegida.
        - degree_centrality: Centralidad de grado.
        - betweenness_centrality: Centralidad de intermediación.
        - closeness_centrality: Centralidad de cercanía.
        - eigenvector_centrality: Centralidad de vector propio.
    """
    """
    Poner el modelo en modo evaluación para desactivar 
    dropout(técnica de regularización) y batchnorm (normaliza la media y varianza de las activaciones 
    por minibatch para estabilizar y acelerar el entrenamiento.)
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, edge_index) # Ejecución hacia adelante para obtener logits y embeddings ocultos
        probs = F.softmax(logits, dim=1) # Convertir logits a probabilidades con softmax
    # Calcular métricas de centralidad con networkx
    centrality_metrics = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G)
    }
    # Recopilar datos por nodo
    analysis_data = [{
        'node': i,
        'degree': G.degree[i],
        'predicted_class': logits[i].argmax().item(),
        'confidence': probs[i].max().item(),
        'degree_centrality': centrality_metrics['degree'][i],
        'betweenness_centrality': centrality_metrics['betweenness'][i],
        'closeness_centrality': centrality_metrics['closeness'][i],
        'eigenvector_centrality': centrality_metrics['eigenvector'][i],
    } for i in range(len(G))]

    return pd.DataFrame(analysis_data) # Devolver un DataFrame con los resultados