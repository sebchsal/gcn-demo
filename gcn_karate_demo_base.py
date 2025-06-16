import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

"""
Prompt utilizado para la extracción de la base de la demo
“Ahora dame otro ejemplo sencillo para realizar una demo sencilla enfocado en 
las GCN siendo esta una clasificación de nodos, dame las fuentes para basarme en el ejemplo.”
Generado por Claude.ai 
"""
class SimpleGCN(nn.Module):
    """
    GCN simplificada para el Karate Club
    Arquitectura minimalista para máxima interpretabilidad
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        # Primera capa: características -> representación oculta
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        
        # Segunda capa: representación oculta -> clases
        h2 = self.conv2(h1, edge_index)
        
        return h2, h1  # Retornamos también las embeddings intermedias

def create_karate_dataset():
    """
    Crea el dataset del Karate Club de Zachary
    - 34 nodos (miembros del club)
    - 78 aristas (amistades/interacciones)  
    - 2 clases (facción de Mr. Hi vs Oficial John A.)
    - Features: grado del nodo como característica única
    """
    # Crear el grafo de Karate Club
    G = nx.karate_club_graph()
    
    # Convertir a tensores de PyTorch Geometric
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # Hacer no dirigido
    
    # Usar el grado como característica (muy interpretable)
    degrees = dict(G.degree())
    x = torch.tensor([[degrees[i]] for i in range(len(G))], dtype=torch.float)
    
    # Labels: club original (0) vs club separado (1)
    # Basado en la división histórica real del club
    y = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(len(G))], dtype=torch.long)
    
    # Crear máscaras de entrenamiento (algunos nodos de cada clase)
    train_mask = torch.zeros(len(G), dtype=torch.bool)
    # Entrenar con el líder de cada facción + algunos miembros representativos
    train_nodes = [0, 33, 1, 2, 3, 8, 13, 19, 20, 22]  # Mix de ambas clases
    train_mask[train_nodes] = True
    
    # El resto para test
    test_mask = ~train_mask
    
    print(f"Dataset Karate Club:")
    print(f"Nodos: {len(G)}")
    print(f"Aristas: {G.number_of_edges()}")
    print(f"Características por nodo: {x.shape[1]}")
    print(f"Clases: 2 (Mr. Hi=0, Officer John=1)")
    print(f"Distribución de clases: {y.sum().item()}/{len(y)-y.sum().item()}")
    print(f"Nodos de entrenamiento: {train_mask.sum()}")
    print(f"Nodos de test: {test_mask.sum()}")
    
    return x, edge_index, y, train_mask, test_mask, G

def visualize_graph(G, y, title="Karate Club Graph", predictions=None, node_embeddings=None):
    """Visualiza el grafo con colores por clase"""
    plt.figure(figsize=(12, 8))
    
    # Layout del grafo
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # Colores por clase real
    colors_true = ['lightblue' if label == 0 else 'lightcoral' for label in y]
    
    if predictions is not None:
        # Crear subplot para comparación
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Grafo con etiquetas verdaderas
        nx.draw(G, pos, node_color=colors_true, with_labels=True, 
                node_size=500, font_size=8, font_weight='bold', ax=ax1)
        ax1.set_title("Etiquetas Verdaderas")
        ax1.legend(['Mr. Hi (0)', 'Officer John (1)'])
        
        # Grafo con predicciones
        colors_pred = ['lightblue' if pred == 0 else 'lightcoral' for pred in predictions]
        # Marcar errores con borde rojo
        edge_colors = ['red' if y[i] != predictions[i] else 'black' for i in range(len(y))]
        
        nx.draw(G, pos, node_color=colors_pred, with_labels=True,
                node_size=500, font_size=8, font_weight='bold', ax=ax2,
                edgecolors=edge_colors, linewidths=2)
        ax2.set_title("Predicciones del Modelo")
        ax2.legend(['Predicción: Mr. Hi (0)', 'Predicción: Officer John (1)'])
        
    else:
        # Solo grafo original
        nx.draw(G, pos, node_color=colors_true, with_labels=True,
                node_size=500, font_size=10, font_weight='bold')
        plt.title(title)
        
        # Leyenda
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='lightblue', label='Mr. Hi (0)')
        red_patch = mpatches.Patch(color='lightcoral', label='Officer John (1)')
        plt.legend(handles=[blue_patch, red_patch])
    
    plt.tight_layout()
    plt.show()

def visualize_embeddings_evolution(embeddings_history, y, epochs_to_show=[0, 10, 50, 100]):
    """Visualiza cómo evolucionan las embeddings durante el entrenamiento"""
    fig, axes = plt.subplots(1, len(epochs_to_show), figsize=(20, 4))
    
    colors = ['blue' if label == 0 else 'red' for label in y]
    
    for idx, epoch in enumerate(epochs_to_show):
        if epoch < len(embeddings_history):
            embeddings = embeddings_history[epoch]
            
            # Si las embeddings tienen más de 2 dimensiones, usar PCA
            if embeddings.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
            else:
                embeddings_2d = embeddings
            
            axes[idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)
            axes[idx].set_title(f'Época {epoch}')
            axes[idx].grid(True, alpha=0.3)
            
            # Añadir números de nodos
            for i, (x, y_coord) in enumerate(embeddings_2d):
                axes[idx].annotate(str(i), (x, y_coord), fontsize=8, alpha=0.7)
    
    plt.suptitle('Evolución de las Embeddings Durante el Entrenamiento')
    plt.tight_layout()
    plt.show()

def analyze_node_importance(G, model, x, edge_index):
    """Analiza la importancia de cada nodo en la clasificación"""
    model.eval()
    
    # Obtener embeddings finales
    with torch.no_grad():
        logits, embeddings = model(x, edge_index)
        probs = F.softmax(logits, dim=1)
    
    # Calcular métricas de centralidad
    centrality_metrics = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G)
    }
    
    # Crear DataFrame para análisis
    import pandas as pd
    
    analysis_data = []
    for i in range(len(G)):
        analysis_data.append({
            'node': i,
            'degree': dict(G.degree())[i],
            'predicted_class': logits[i].argmax().item(),
            'confidence': probs[i].max().item(),
            'degree_centrality': centrality_metrics['degree'][i],
            'betweenness_centrality': centrality_metrics['betweenness'][i],
            'closeness_centrality': centrality_metrics['closeness'][i],
            'eigenvector_centrality': centrality_metrics['eigenvector'][i]
        })
    
    df = pd.DataFrame(analysis_data)
    
    # Visualizar correlaciones
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(df['degree'], df['confidence'], c=df['predicted_class'], alpha=0.7)
    plt.xlabel('Grado del Nodo')
    plt.ylabel('Confianza de Predicción')
    plt.title('Grado vs Confianza')
    plt.colorbar(label='Clase Predicha')
    
    plt.subplot(1, 3, 2)
    plt.scatter(df['betweenness_centrality'], df['confidence'], c=df['predicted_class'], alpha=0.7)
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Confianza de Predicción')
    plt.title('Centralidad vs Confianza')
    plt.colorbar(label='Clase Predicha')
    
    plt.subplot(1, 3, 3)
    # Heatmap de correlaciones
    corr_data = df[['degree', 'confidence', 'degree_centrality', 'betweenness_centrality']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlaciones entre Métricas')
    
    plt.tight_layout()
    plt.show()
    
    return df

def main():
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar datos
    x, edge_index, y, train_mask, test_mask, G = create_karate_dataset()
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask, test_mask = train_mask.to(device), test_mask.to(device)
    
    # Visualizar grafo original
    visualize_graph(G, y.cpu(), "Karate Club - Etiquetas Verdaderas")
    
    # Definir modelo
    model = SimpleGCN(input_dim=1, hidden_dim=4, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nModelo SimpleGCN:")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    print(model)
    
    # Entrenamiento con seguimiento de embeddings
    print("\nIniciando entrenamiento...")
    train_losses = []
    train_accuracies = []
    embeddings_history = []
    
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        
        logits, embeddings = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Calcular precisión
        with torch.no_grad():
            pred = logits[train_mask].argmax(dim=1)
            acc = (pred == y[train_mask]).float().mean()
        
        train_losses.append(loss.item())
        train_accuracies.append(acc.item())
        
        # Guardar embeddings para visualización
        if epoch % 10 == 0:
            embeddings_history.append(embeddings.detach().cpu().numpy())
        
        if epoch % 30 == 0:
            print(f'Época {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}')
    
    # Evaluación final
    model.eval()
    with torch.no_grad():
        logits, final_embeddings = model(x, edge_index)
        test_pred = logits[test_mask].argmax(dim=1)
        test_acc = (test_pred == y[test_mask]).float().mean()
        
        # Predicciones para todo el grafo
        all_pred = logits.argmax(dim=1)
    
    print(f"\nPrecisión en test: {test_acc:.4f}")
    
    # Reporte detallado
    print("\nReporte de clasificación:")
    print(classification_report(y.cpu(), all_pred.cpu(), target_names=['Mr. Hi', 'Officer John']))
    
    # Matriz de confusión
    cm = confusion_matrix(y.cpu(), all_pred.cpu())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Mr. Hi', 'Officer John'],
                yticklabels=['Mr. Hi', 'Officer John'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción')
    plt.show()
    
    # Visualizar resultados
    visualize_graph(G, y.cpu(), predictions=all_pred.cpu())
    
    # Gráficos de entrenamiento
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Pérdida de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies)
    plt.title('Precisión de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Distribución de grados vs clases
    degrees = [G.degree(i) for i in range(len(G))]
    plt.scatter(degrees, y.cpu(), c=y.cpu(), alpha=0.7)
    plt.xlabel('Grado del Nodo')
    plt.ylabel('Clase')
    plt.title('Grado vs Clase')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Visualizar evolución de embeddings
    visualize_embeddings_evolution(embeddings_history, y.cpu())
    
    # Análisis de importancia de nodos
    node_analysis = analyze_node_importance(G, model, x, edge_index)
    
    # Mostrar nodos más importantes
    print("\nNodos más importantes (por confianza):")
    top_nodes = node_analysis.nlargest(5, 'confidence')
    print(top_nodes[['node', 'degree', 'predicted_class', 'confidence']].to_string(index=False))
    
    print("\nNodos menos seguros:")
    uncertain_nodes = node_analysis.nsmallest(5, 'confidence')
    print(uncertain_nodes[['node', 'degree', 'predicted_class', 'confidence']].to_string(index=False))

if __name__ == "__main__":
    main()

# INSTRUCCIONES DE USO:
# 1. Instalar dependencias:
#    pip install torch torch-geometric matplotlib networkx seaborn pandas scikit-learn
#
# 2. Ejecutar el script:
#    python karate_gcn.py
#
# VENTAJAS DE ESTE EJEMPLO:
# - Solo 34 nodos (muy manejable visualmente)
# - Historia real conocida (división del club)
# - Clasificación binaria (más simple)
# - Feature simple (grado del nodo)
# - Resultados 100% interpretables
# - Visualización completa del grafo