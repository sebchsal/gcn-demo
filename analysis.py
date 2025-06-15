import torch          
import torch.nn.functional as F
import pandas as pd
import networkx as nx

def analyze_node_importance(G, model, x, edge_index):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, edge_index)
        probs = F.softmax(logits, dim=1)

    centrality_metrics = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G)
    }

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

    return pd.DataFrame(analysis_data)