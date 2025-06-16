import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.metrics import confusion_matrix
#Aporte de funcion de visualizacion con Plotly
def create_comprehensive_dashboard(train_losses, train_accuracies, node_analysis, y, predictions, G):
    """Genera un panel con múltiples análisis:
        1) Curva de pérdida de entrenamiento
        2) Curva de precisión de entrenamiento
        3) Mapa de calor de la matriz de confusión
        4) Histogramas de confianza por clase
        5) Dispersión centralidad vs confianza
        6) Dispersión grado vs confianza
        7) Tabla: Top 5 nodos con mayor confianza
        8) Tabla: Top 5 nodos con menor confianza
        9) Histograma de distribución de grados

    Args:
        - train_losses (list): Pérdida por época.
        - train_accuracies (list): Precisión por época.
        - node_analysis (pd.DataFrame): Métricas a nivel de nodo.
        - y (np.ndarray): Etiquetas verdaderas.
        - predictions (np.ndarray): Etiquetas predichas.
        - G (networkx.Graph): Grafo para la distribución de grados.

    Retorna:
        - fig (plotly.graph_objs.Figure): Figura del dashboard.
    """
    # Crear cuadrícula 3x3 de subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Pérdida de Entrenamiento', 'Precisión de Entrenamiento', 'Matriz de Confusión',
            'Confianza por Clase', 'Centralidad vs Confianza', 'Grado vs Confianza',
            'Top Nodos Seguros', 'Top Nodos Inseguros', 'Distribución de Grados'
        ),
        specs=[
            [{}, {}, {"type": "heatmap"}],
            [{}, {}, {}],
            [{"type": "table"}, {"type": "table"}, {}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    # Panel 1: Curva de pérdida
    epochs = list(range(len(train_losses)))
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name='Loss', line=dict(color='red')), row=1, col=1)
    # Panel 2: Curva de precisión
    fig.add_trace(go.Scatter(x=epochs, y=train_accuracies, name='Accuracy', line=dict(color='green')), row=1, col=2)
    # Panel 3: Matriz de confusión normalizada
    cm = confusion_matrix(y, predictions)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig.add_trace(go.Heatmap(z=cm_norm, x=['Mr. Hi', 'Officer John'], y=['Mr. Hi', 'Officer John'], colorscale='Blues'), row=1, col=3)
    # Panel 4: Histogramas de confianza
    conf_0 = node_analysis[node_analysis['predicted_class'] == 0]['confidence']
    conf_1 = node_analysis[node_analysis['predicted_class'] == 1]['confidence']
    if not conf_0.empty:
        fig.add_trace(go.Histogram(x=conf_0, name='Mr. Hi', marker_color='blue'), row=2, col=1)
    if not conf_1.empty:
        fig.add_trace(go.Histogram(x=conf_1, name='Officer John', marker_color='red'), row=2, col=1)
    # Panel 5: Centralidad de intermediación vs confianza
    fig.add_trace(go.Scatter(
        x=node_analysis['betweenness_centrality'], y=node_analysis['confidence'],
        mode='markers', marker=dict(size=12, color=node_analysis['predicted_class'], colorscale='Viridis'),
        text=node_analysis['node']
    ), row=2, col=2)
    # Panel 6: Grado vs confianza
    fig.add_trace(go.Scatter(
        x=node_analysis['degree'], y=node_analysis['confidence'],
        mode='markers', marker=dict(size=12, color=node_analysis['predicted_class'], colorscale='Cividis'),
        text=node_analysis['node']
    ), row=2, col=3)
    # Panel 7: Tabla de top 5 nodos confiables
    top_conf = node_analysis.nlargest(5, 'confidence')
    fig.add_trace(go.Table(
        header=dict(values=['Nodo', 'Clase', 'Confianza', 'Grado']),
        cells=dict(values=[
            top_conf['node'],
            top_conf['predicted_class'],
            top_conf['confidence'].round(3),
            top_conf['degree']
        ])
    ), row=3, col=1)
    # Panel 8: Tabla de top 5 nodos menos confiables
    low_conf = node_analysis.nsmallest(5, 'confidence')
    fig.add_trace(go.Table(
        header=dict(values=['Nodo', 'Clase', 'Confianza', 'Grado']),
        cells=dict(values=[
            low_conf['node'],
            low_conf['predicted_class'],
            low_conf['confidence'].round(3),
            low_conf['degree']
        ])
    ), row=3, col=2)
    # Panel 9: Histograma de distribución de grados
    degrees = [G.degree(i) for i in G.nodes()]
    fig.add_trace(go.Histogram(x=degrees, nbinsx=10, marker_color='purple'), row=3, col=3)

    fig.update_layout(height=1200, width=1000, title_text="Dashboard de Análisis del GCN - Karate Club", title_x=0.5)
    return fig
