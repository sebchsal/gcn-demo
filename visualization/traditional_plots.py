import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

#Funcion original del codigo base mejorada con Plotly
def visualize_graph(G, y, title="Karate Club Graph", predictions=None):
    """Visualización estática con Matplotlib del grafo.
    Muestra las etiquetas verdaderas y opcionalmente las predicciones.
    """
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    colors_true = ['lightblue' if label==0 else 'lightcoral' for label in y]

    if predictions is not None:
        # Dibujar etiquetas verdaderas y predicciones lado a lado
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
        nx.draw(G, pos, node_color=colors_true, with_labels=True, node_size=500, ax=ax1)
        ax1.set_title("Etiquetas Verdaderas")
        colors_pred = ['lightblue' if p==0 else 'lightcoral' for p in predictions]
        nx.draw(G, pos, node_color=colors_pred, with_labels=True, node_size=500, ax=ax2)
        ax2.set_title("Predicciones del Modelo")
    else:
        # Solo etiquetas verdaderas
        fig = plt.figure(figsize=(12,8))
        nx.draw(G, pos, node_color=colors_true, with_labels=True, node_size=500)
        plt.title(title)

    plt.tight_layout()
    plt.show()
    return fig
#Funcion original del codigo base mejorada con Plotly
def visualize_embeddings_evolution(embeddings_history, y, epochs_to_show=[0,10,50,100]):
    """Muestra la evolución de los embeddings en épocas seleccionadas 
    usando PCA si son de más de 2 dimensiones."""
    fig, axes = plt.subplots(1, len(epochs_to_show), figsize=(20,4))
    colors = ['blue' if label==0 else 'red' for label in y]

    for idx, epoch in enumerate(epochs_to_show):
        if epoch < len(embeddings_history):
            emb = embeddings_history[epoch]
            # Reducir a 2D si es necesario
            if emb.shape[1] > 2:
                pca = PCA(n_components=2)
                emb2d = pca.fit_transform(emb)
            else:
                emb2d = emb
            ax = axes[idx]
            ax.scatter(emb2d[:,0], emb2d[:,1], c=colors, alpha=0.7)
            ax.set_title(f'Época {epoch}')
            ax.grid(True, alpha=0.3)
            for i, (x_c, y_c) in enumerate(emb2d):
                ax.annotate(str(i), (x_c, y_c), fontsize=8, alpha=0.7)

    fig.suptitle('Evolución de las Embeddings Durante el Entrenamiento')
    plt.tight_layout()
    plt.show()
    return fig