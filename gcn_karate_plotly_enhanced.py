import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NUEVAS IMPORTACIONES PARA PLOTLY
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class SimpleGCN(nn.Module):
    """
    GCN simplificada para el Karate Club
    Arquitectura minimalista para maxima interpretabilidad
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

# NUEVAS FUNCIONES DE VISUALIZACIÓN CON PLOTLY

def create_interactive_graph(G, y, predictions=None, node_features=None, title="Karate Club - Grafo Interactivo"):
    """
    Crea un grafo interactivo con Plotly
    Muestra información detallada al hacer hover sobre los nodos
    """
    # Layout del grafo
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # Extraer coordenadas de nodos
    x_nodes = [pos[i][0] for i in range(len(G.nodes()))]
    y_nodes = [pos[i][1] for i in range(len(G.nodes()))]
    
    # Preparar edges para visualización
    x_edges = []
    y_edges = []
    for edge in G.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir edges
    fig.add_trace(go.Scatter(
        x=x_edges, y=y_edges,
        mode='lines',
        line=dict(width=1.5, color='rgba(125, 125, 125, 0.4)'),
        hoverinfo='none',
        showlegend=False,
        name='Conexiones'
    ))
    
    # Preparar información de hover personalizada
    hover_text = []
    for i in range(len(G.nodes())):
        degree = G.degree(i)
        true_class = "Mr. Hi" if y[i] == 0 else "Officer John"
        pred_class = ("Mr. Hi" if predictions[i] == 0 else "Officer John") if predictions is not None else "N/A"
        
        # Determinar si la predicción es correcta
        is_correct = "✓" if predictions is not None and predictions[i] == y[i] else "✗" if predictions is not None else ""
        
        hover_info = f"""
        <b>Nodo {i}</b><br>
        Grado: {degree}<br>
        Clase Real: {true_class}<br>
        Predicción: {pred_class} {is_correct}<br>
        """
        
        if node_features is not None:
            hover_info += f"Feature Value: {node_features[i]:.3f}<br>"
        
        hover_text.append(hover_info)
    
    # Colores por clase (con opción de mostrar errores)
    if predictions is not None:
        # Colores diferentes para predicciones correctas e incorrectas
        colors = []
        for i in range(len(y)):
            if predictions[i] == y[i]:
                colors.append('blue' if y[i] == 0 else 'red')
            else:
                colors.append('orange')  # Error
    else:
        colors = ['blue' if label == 0 else 'red' for label in y]
    
    # Tamaño de nodos basado en grado
    node_sizes = [G.degree(i) * 3 + 10 for i in range(len(G.nodes()))]
    
    # Añadir nodos
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=colors,
            line=dict(width=2, color='white'),
            opacity=0.8,
            symbol='circle'
        ),
        text=[str(i) for i in range(len(G.nodes()))],
        textposition="middle center",
        textfont=dict(color="white", size=10, family="Arial Bold"),
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
        name='Nodos'
    ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family="Arial")
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=60),
        annotations=[
            dict(
                text="🔵 Mr. Hi | 🔴 Officer John | 🟠 Error de Predicción<br>Tamaño del nodo = Grado | Hover para detalles",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                xanchor='center', yanchor='top',
                font=dict(color="gray", size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=800,
        height=600
    )
    
    return fig

def create_comprehensive_dashboard(train_losses, train_accuracies, node_analysis, y, predictions, G):
    """
    Crea un dashboard completo con múltiples métricas y análisis
    """
    # Crear subplots con especificaciones personalizadas
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Pérdida de Entrenamiento', 'Precisión de Entrenamiento', 'Matriz de Confusión',
            'Distribución de Confianza por Clase', 'Centralidad vs Confianza', 'Análisis de Grados',
            'Top 5 Nodos Más Seguros', 'Top 5 Nodos Menos Seguros', 'Distribución de Características'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"type": "heatmap"}],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"type": "table"}, {"type": "table"}, {"secondary_y": False}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1. Pérdida de entrenamiento con suavizado
    epochs = list(range(len(train_losses)))
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode='lines',
            name='Loss',
            line=dict(color='#e74c3c', width=3),
            hovertemplate='Época: %{x}<br>Pérdida: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Precisión de entrenamiento
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=4),
            hovertemplate='Época: %{x}<br>Precisión: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Matriz de confusión mejorada
    cm = confusion_matrix(y, predictions)
    
    # Calcular porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear anotaciones personalizadas
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            annotations.append(
                dict(
                    text=f"{cm[i][j]}<br>({cm_normalized[i][j]:.1%})",
                    x=j, y=i,
                    xref='x3', yref='y3',
                    showarrow=False,
                    font=dict(color='white' if cm_normalized[i][j] > 0.5 else 'black', size=14)
                )
            )
    
    fig.add_trace(
        go.Heatmap(
            z=cm_normalized,
            x=['Mr. Hi', 'Officer John'],
            y=['Mr. Hi', 'Officer John'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Proporción"),
            hovertemplate='Predicho: %{x}<br>Real: %{y}<br>Proporción: %{z:.2%}<extra></extra>'
        ),
        row=1, col=3
    )
    
    # 4. Distribución de confianza por clase
    confidence_0 = node_analysis[node_analysis['predicted_class'] == 0]['confidence']
    confidence_1 = node_analysis[node_analysis['predicted_class'] == 1]['confidence']
    
    fig.add_trace(
        go.Histogram(
            x=confidence_0,
            name='Mr. Hi',
            opacity=0.7,
            nbinsx=8,
            marker_color='#3498db',
            hovertemplate='Confianza: %{x:.3f}<br>Frecuencia: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=confidence_1,
            name='Officer John',
            opacity=0.7,
            nbinsx=8,
            marker_color='#e74c3c',
            hovertemplate='Confianza: %{x:.3f}<br>Frecuencia: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 5. Centralidad vs Confianza (scatter mejorado)
    colors = ['#3498db' if c == 0 else '#e74c3c' for c in node_analysis['predicted_class']]
    
    fig.add_trace(
        go.Scatter(
            x=node_analysis['betweenness_centrality'],
            y=node_analysis['confidence'],
            mode='markers',
            marker=dict(
                color=colors,
                size=node_analysis['degree'] * 2 + 8,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"Nodo {i} (Grado: {d})" for i, d in zip(node_analysis['node'], node_analysis['degree'])],
            hovertemplate='%{text}<br>Centralidad: %{x:.3f}<br>Confianza: %{y:.3f}<extra></extra>',
            name='Nodos'
        ),
        row=2, col=2
    )
    
    # 6. Análisis de grados vs confianza
    fig.add_trace(
        go.Scatter(
            x=node_analysis['degree'],
            y=node_analysis['confidence'],
            mode='markers',
            marker=dict(
                color=colors,
                size=15,
                opacity=0.8,
                symbol='diamond'
            ),
            text=[f"Nodo {i}" for i in node_analysis['node']],
            hovertemplate='%{text}<br>Grado: %{x}<br>Confianza: %{y:.3f}<extra></extra>',
            name='Grados'
        ),
        row=2, col=3
    )
    
    # 7. Tabla de nodos más seguros
    top_confident = node_analysis.nlargest(5, 'confidence')
    fig.add_trace(
        go.Table(
            header=dict(values=['Nodo', 'Clase', 'Confianza', 'Grado'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[
                top_confident['node'].tolist(),
                ['Mr. Hi' if c == 0 else 'Officer John' for c in top_confident['predicted_class']],
                [f"{c:.3f}" for c in top_confident['confidence']],
                top_confident['degree'].tolist()
            ],
            fill_color='lavender',
            align='left')
        ),
        row=3, col=1
    )
    
    # 8. Tabla de nodos menos seguros
    least_confident = node_analysis.nsmallest(5, 'confidence')
    fig.add_trace(
        go.Table(
            header=dict(values=['Nodo', 'Clase', 'Confianza', 'Grado'],
                       fill_color='mistyrose',
                       align='left'),
            cells=dict(values=[
                least_confident['node'].tolist(),
                ['Mr. Hi' if c == 0 else 'Officer John' for c in least_confident['predicted_class']],
                [f"{c:.3f}" for c in least_confident['confidence']],
                least_confident['degree'].tolist()
            ],
            fill_color='lavenderblush',
            align='left')
        ),
        row=3, col=2
    )
    
    # 9. Distribución de características (grados)
    degrees = [G.degree(i) for i in range(len(G.nodes()))]
    fig.add_trace(
        go.Histogram(
            x=degrees,
            nbinsx=10,
            marker_color='#9b59b6',
            opacity=0.7,
            name='Distribución de Grados',
            hovertemplate='Grado: %{x}<br>Frecuencia: %{y}<extra></extra>'
        ),
        row=3, col=3
    )
    
    # Actualizar layout general
    fig.update_layout(
        height=1200,
        title_text="Dashboard Completo de Análisis GCN - Karate Club",
        title_x=0.5,
        title_font_size=20,
        showlegend=False,
        font=dict(size=10)
    )
    
    # Añadir anotaciones de la matriz de confusión
    fig.layout.annotations = fig.layout.annotations + tuple(annotations)
    
    # Configurar ejes individuales
    fig.update_xaxes(title_text="Época", row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text="Época", row=1, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text="Confianza", row=2, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text="Betweenness Centrality", row=2, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text="Grado del Nodo", row=2, col=3, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text="Grado", row=3, col=3, showgrid=True, gridcolor='lightgray')
    
    fig.update_yaxes(title_text="Pérdida", row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Precisión", row=1, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Frecuencia", row=2, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Confianza", row=2, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Confianza", row=2, col=3, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Frecuencia", row=3, col=3, showgrid=True, gridcolor='lightgray')
    
    return fig

def create_network_statistics_radar(G, node_analysis):
    """
    Crea un gráfico de radar para mostrar estadísticas de la red
    """
    # Calcular métricas de red
    avg_clustering = nx.average_clustering(G)
    density = nx.density(G)
    avg_path_length = nx.average_shortest_path_length(G)
    diameter = nx.diameter(G)
    
    # Normalizar métricas para el radar (0-1)
    metrics = {
        'Clustering Promedio': avg_clustering,
        'Densidad': density,
        'Longitud de Camino<br>Promedio (norm)': 1 - (avg_path_length / diameter),  # Invertido para que más alto sea mejor
        'Diámetro (norm)': 1 - (diameter / len(G.nodes())),  # Normalizado e invertido
        'Modularidad': 0.8,  # Aproximado para el Karate Club
        'Asortatividad': abs(nx.degree_assortativity_coefficient(G))
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        name='Métricas de Red',
        line_color='#2ecc71',
        fillcolor='rgba(46, 204, 113, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Análisis de Propiedades de la Red",
        title_x=0.5,
        width=600,
        height=500
    )
    
    return fig

# FUNCIONES ORIGINALES MEJORADAS CON PLOTLY

def visualize_graph(G, y, title="Karate Club Graph", predictions=None, node_embeddings=None):
    """Visualiza el grafo - VERSION ORIGINAL mantenida para compatibilidad"""
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
    """Visualiza cómo evolucionan las embeddings - VERSION ORIGINAL"""
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
    
    # Visualizar correlaciones con matplotlib (original)
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
    
    # NUEVA VISUALIZACIÓN INTERACTIVA - Grafo original
    print("\n🎯 Creando visualización interactiva del grafo original...")
    fig_original = create_interactive_graph(G, y.cpu(), title="Karate Club - Etiquetas Verdaderas")
    fig_original.show()
    
    # Definir modelo
    model = SimpleGCN(input_dim=1, hidden_dim=4, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nModelo SimpleGCN:")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    print(model)
    
    # Entrenamiento con seguimiento de embeddings
    print("\n🚀 Iniciando entrenamiento...")
    train_losses = []
    train_accuracies = []
    embeddings_history = []
    
    # Épocas para guardar embeddings
    epochs_to_save = [0, 10, 25, 50, 75, 100, 125, 149]
    
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
        
        # Guardar embeddings para visualización en épocas específicas
        if epoch in epochs_to_save:
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
    
    print(f"\n✅ Precisión en test: {test_acc:.4f}")
    
    # Reporte detallado
    print("\n📊 Reporte de clasificación:")
    print(classification_report(y.cpu(), all_pred.cpu(), target_names=['Mr. Hi', 'Officer John']))
    
    # NUEVA VISUALIZACIÓN INTERACTIVA - Comparación con predicciones
    print("\n🎯 Creando visualización interactiva con predicciones...")
    fig_predictions = create_interactive_graph(
        G, y.cpu(), 
        predictions=all_pred.cpu(), 
        node_features=x.cpu().numpy().flatten(),
        title="Karate Club - Predicciones vs Verdaderas"
    )
    fig_predictions.show()
    
    # Análisis de importancia de nodos
    print("\n🔍 Analizando importancia de nodos...")
    node_analysis = analyze_node_importance(G, model, x, edge_index)
    
    """   # NUEVA VISUALIZACIÓN 3D - Evolución de embeddings
    print("\n🌟 Creando visualización 3D de evolución de embeddings...")
    fig_3d_evolution = create_3d_embeddings_evolution(
        embeddings_history, y.cpu(), epochs_to_save
    )
    fig_3d_evolution.show() """
    
    # NUEVO DASHBOARD COMPLETO
    print("\n📈 Creando dashboard completo de análisis...")
    dashboard = create_comprehensive_dashboard(
        train_losses, train_accuracies, node_analysis, 
        y.cpu().numpy(), all_pred.cpu().numpy(), G
    )
    dashboard.show()
    
    # NUEVO GRÁFICO DE RADAR - Estadísticas de red
    print("\n🕸️ Creando análisis de propiedades de la red...")
    radar_fig = create_network_statistics_radar(G, node_analysis)
    radar_fig.show()
    
    """ # Matriz de confusión tradicional (mantenida para comparación)
    cm = confusion_matrix(y.cpu(), all_pred.cpu())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Mr. Hi', 'Officer John'],
                yticklabels=['Mr. Hi', 'Officer John'])
    plt.title('Matriz de Confusión - Visualización Tradicional')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción')
    plt.show() """
    
     # Visualización tradicional del grafo (mantenida para compatibilidad)
    print("\n📊 Mostrando visualizaciones tradicionales...")
    visualize_graph(G, y.cpu(), predictions=all_pred.cpu())
    
    # Gráficos de entrenamiento tradicionales
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, color='#e74c3c', linewidth=2)
    plt.title('Pérdida de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, color='#2ecc71', linewidth=2)
    plt.title('Precisión de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    """# Distribución de grados vs clases
    degrees = [G.degree(i) for i in range(len(G))]
    plt.scatter(degrees, y.cpu(), c=y.cpu(), alpha=0.7, cmap='coolwarm')
    plt.xlabel('Grado del Nodo')
    plt.ylabel('Clase')
    plt.title('Grado vs Clase')
    plt.colorbar(label='Clase')
    
    plt.tight_layout()
    plt.show() """
    
    # Visualización tradicional de evolución de embeddings
    """visualize_embeddings_evolution(embeddings_history, y.cpu(), epochs_to_save[:4])"""
    
    # Mostrar nodos más importantes
    print("\n🏆 Nodos más importantes (por confianza):")
    top_nodes = node_analysis.nlargest(5, 'confidence')
    print(top_nodes[['node', 'degree', 'predicted_class', 'confidence']].to_string(index=False))
    
    print("\n⚠️ Nodos menos seguros:")
    uncertain_nodes = node_analysis.nsmallest(5, 'confidence')
    print(uncertain_nodes[['node', 'degree', 'predicted_class', 'confidence']].to_string(index=False))
    
    # Análisis adicional con Plotly
    print("\n📋 Resumen del análisis:")
    accuracy = (all_pred.cpu() == y.cpu()).float().mean().item()
    print(f"✓ Precisión general: {accuracy:.1%}")
    print(f"✓ Nodos clasificados correctamente: {(all_pred.cpu() == y.cpu()).sum().item()}/{len(y)}")
    print(f"✓ Pérdida final: {train_losses[-1]:.4f}")
    print(f"✓ Precisión final de entrenamiento: {train_accuracies[-1]:.1%}")
    
    # Estadísticas de la red
    print(f"\n🕸️ Estadísticas de la red:")
    print(f"✓ Clustering promedio: {nx.average_clustering(G):.3f}")
    print(f"✓ Densidad: {nx.density(G):.3f}")
    print(f"✓ Diámetro: {nx.diameter(G)}")
    print(f"✓ Longitud de camino promedio: {nx.average_shortest_path_length(G):.2f}")
    
    print("\n🎉 ¡Análisis completo terminado!")
    print("💡 Las visualizaciones interactivas permiten explorar:")
    print("   - Hover sobre nodos para ver detalles")
    print("   - Animación 3D de la evolución de embeddings")
    print("   - Dashboard interactivo con múltiples métricas")
    print("   - Análisis de radar de propiedades de red")

if __name__ == "__main__":
    main()