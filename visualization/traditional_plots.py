import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

def visualize_graph(G, y, title="Karate Club Graph", predictions=None):
    """
    EN: Static Matplotlib drawing of the graph, with true labels and optional model predictions.
    ES: Dibujo estático con Matplotlib del grafo, con etiquetas reales y predicciones opcionales.
    """
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    colors_true = ['lightblue' if label==0 else 'lightcoral' for label in y]

    if predictions is not None:
        # EN: Two‐pane figure: real vs. predicted
        # ES: Figura de dos paneles: reales vs. predichas
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
        nx.draw(G, pos, node_color=colors_true, with_labels=True, node_size=500, ax=ax1)
        ax1.set_title("Etiquetas Verdaderas")
        colors_pred = ['lightblue' if p==0 else 'lightcoral' for p in predictions]
        nx.draw(G, pos, node_color=colors_pred, with_labels=True, node_size=500, ax=ax2)
        ax2.set_title("Predicciones del Modelo")
    else:
        # EN: Single figure: only true labels
        # ES: Única figura: solo etiquetas reales
        fig = plt.figure(figsize=(12,8))
        nx.draw(G, pos, node_color=colors_true, with_labels=True, node_size=500)
        plt.title(title)

    plt.tight_layout()
    plt.show()
    return fig

def visualize_embeddings_evolution(embeddings_history, y, epochs_to_show=[0,10,50,100]):
    """
    EN: Plots snapshots of embeddings across epochs using PCA if needed.
    ES: Dibuja instantáneas de embeddings en distintas épocas, aplicando PCA si es necesario.
    """
    fig, axes = plt.subplots(1, len(epochs_to_show), figsize=(20,4))
    colors = ['blue' if label==0 else 'red' for label in y]

    for idx, epoch in enumerate(epochs_to_show):
        if epoch < len(embeddings_history):
            emb = embeddings_history[epoch]
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