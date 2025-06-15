import plotly.graph_objects as go
import networkx as nx

def create_interactive_graph(G, y, predictions=None, node_features=None, title="Karate Club - Grafo Interactivo"):
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    x_nodes = [pos[i][0] for i in G.nodes()]
    y_nodes = [pos[i][1] for i in G.nodes()]

    x_edges, y_edges = [], []
    for edge in G.edges():
        x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
        y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_edges, y=y_edges, mode='lines',
        line=dict(width=1.5, color='rgba(125, 125, 125, 0.4)'),
        hoverinfo='none', showlegend=False
    ))

    hover_text = []
    for i in G.nodes():
        degree = G.degree(i)
        true_class = "Mr. Hi" if y[i] == 0 else "Officer John"
        pred_class = ("Mr. Hi" if predictions[i] == 0 else "Officer John") if predictions is not None else "N/A"
        is_correct = "✓" if predictions is not None and predictions[i] == y[i] else "✗" if predictions is not None else ""
        hover = f"Nodo {i}<br>Grado: {degree}<br>Clase Real: {true_class}<br>Predicción: {pred_class} {is_correct}"
        if node_features is not None:
            hover += f"<br>Feature: {node_features[i]:.3f}"
        hover_text.append(hover)

    colors = []
    for i in G.nodes():
        if predictions is None:
            colors.append('blue' if y[i] == 0 else 'red')
        else:
            if predictions[i] == y[i]:
                colors.append('blue' if y[i] == 0 else 'red')
            else:
                colors.append('orange')

    node_sizes = [G.degree(i) * 3 + 10 for i in G.nodes()]

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(size=node_sizes, color=colors, line=dict(width=2, color='white'), opacity=0.8),
        text=[str(i) for i in G.nodes()], textposition="middle center",
        textfont=dict(color="white", size=10),
        hovertext=hover_text, hovertemplate='%{hovertext}<extra></extra>'
    ))

    fig.update_layout(
        title=title, title_x=0.5,
        showlegend=False,
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white', width=800, height=600
    )

    return fig
