import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

import networkx as nx

import matplotlib.pyplot as plt

from model import SimpleGCN
from data import create_karate_dataset
from training import train
from analysis import analyze_node_importance

from visualization.graph_plotly import create_interactive_graph
from visualization.dashboard import create_comprehensive_dashboard
from visualization.radar_analysis import create_network_statistics_radar
from visualization.traditional_plots import visualize_graph, visualize_embeddings_evolution


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    x, edge_index, y, train_mask, test_mask, G = create_karate_dataset()
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask, test_mask = train_mask.to(device), test_mask.to(device)

    print("\nVisualización original del grafo...")
    create_interactive_graph(G, y.cpu(), title="Karate Club - Etiquetas Verdaderas").show()

    model = SimpleGCN(input_dim=1, hidden_dim=4, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"\nModelo SimpleGCN con {sum(p.numel() for p in model.parameters())} parámetros")

    print("\nEntrenamiento...")
    save_epochs = [0, 10, 50, 100]
    train_losses, train_accuracies, embeddings_history = train(
        model, x, edge_index, y, train_mask, optimizer, criterion,
        epochs=150, save_epochs=save_epochs
    )

    model.eval()
    with torch.no_grad():
        logits, final_embeddings = model(x, edge_index)
        all_pred = logits.argmax(dim=1)
        test_pred = logits[test_mask].argmax(dim=1)
        test_acc = (test_pred == y[test_mask]).float().mean()

    print(f"\nPrecisión en test: {test_acc:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y.cpu(), all_pred.cpu(), target_names=['Mr. Hi', 'Officer John']))

    print("\nVisualización interactiva con predicciones...")
    create_interactive_graph(
        G, y.cpu(), predictions=all_pred.cpu(),
        node_features=x.cpu().numpy().flatten(),
        title="Karate Club - Predicciones vs Verdaderas"
    ).show()

    print("\nAnalizando importancia de nodos...")
    node_analysis = analyze_node_importance(G, model, x, edge_index)

    print("\nDashboard completo...")
    create_comprehensive_dashboard(
        train_losses, train_accuracies, node_analysis,
        y.cpu().numpy(), all_pred.cpu().numpy(), G
    ).show()

    print("\nRadar de propiedades de red...")
    create_network_statistics_radar(G, node_analysis).show()

    # Muestra dibujo estático con Matplotlib del grafo
    print("\nVisualizando grafo con Matplotlib...")
    visualize_graph(G, y.cpu(), predictions=all_pred.cpu())

    # Muestra evolución de embeddings
    print("\nVisualizando evolución de embeddings")
    visualize_embeddings_evolution(embeddings_history,y.cpu().numpy(),
    epochs_to_show=list(range(len(embeddings_history))))

    # Esperar confirmación del usuario
    input("\nPresiona ENTER para continuar...")

    print("\nTop nodos por confianza:")
    print(node_analysis.nlargest(5, 'confidence')[['node', 'degree', 'predicted_class', 'confidence']])

    print("\nNodos menos seguros:")
    print(node_analysis.nsmallest(5, 'confidence')[['node', 'degree', 'predicted_class', 'confidence']])


if __name__ == "__main__":
    main()