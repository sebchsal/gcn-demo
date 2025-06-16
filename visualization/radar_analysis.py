import networkx as nx
import plotly.graph_objects as go
#Aporte de funcion de visualizacion con Plotly
def create_network_statistics_radar(G, node_analysis=None):
    """Crea un diagrama de radar con estadísticas globales de la red:
        - Clustering promedio
        - Densidad
        - Camino promedio normalizado
        - Diámetro normalizado
        - Modularidad (valor de ejemplo)
        - Asortatividad

    Args:
        - G (networkx.Graph): Grafo de análisis.
        - node_analysis (pd.DataFrame, opcional): Métricas por nodo (no usado aquí).

    Retorna:
        - fig (plotly.graph_objs.Figure): Figura del radar.
    """
    avg_clustering = nx.average_clustering(G)
    density = nx.density(G)
    avg_path_length = nx.average_shortest_path_length(G)
    diameter = nx.diameter(G)
    modularity = 0.8  # Valor de ejemplo
    assortativity = abs(nx.degree_assortativity_coefficient(G))

    metrics = {
        'Clustering Promedio': avg_clustering,
        'Densidad': density,
        'Longitud de Camino<br>Promedio (norm)': 1 - (avg_path_length / diameter),
        'Diámetro (norm)': 1 - (diameter / len(G.nodes())),
        'Modularidad': modularity,
        'Asortatividad': assortativity
    }

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        name='Métricas de Red',
        line_color='green',
        fillcolor='rgba(46, 204, 113, 0.3)'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Análisis de Propiedades de la Red",
        title_x=0.5,
        width=600,
        height=500
    )

    return fig
