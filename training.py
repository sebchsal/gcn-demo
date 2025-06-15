import torch
import torch.nn.functional as F
#Funcion de entranamiento extraida del codigo base en la funcion main
def train(model, x, edge_index, y, train_mask, optimizer, criterion, epochs, save_epochs):
    """Bucle de entrenamiento para el modelo GCN, registra pérdida, precisión y embeddings.

    Args:
        - model (torch.nn.Module): Modelo GCN a entrenar.
        - x (torch.Tensor): Características de nodos.
        - edge_index (torch.Tensor): Aristas del grafo.
        - y (torch.Tensor): Etiquetas verdaderas.
        - train_mask (torch.BoolTensor): Máscara para nodos de entrenamiento.
        - optimizer (torch.optim.Optimizer): Optimizador.
        - criterion (callable): Función de pérdida (e.g., CrossEntropyLoss).
        - epochs (int): Épocas de entrenamiento.
        - save_epochs (list[int]): Épocas en las que se guardan embeddings.

    Retorna:
        - train_losses (list[float]): Pérdida por época.
        - train_accuracies (list[float]): Precisión por época.
        - embeddings_history (list[np.ndarray]): Embeddings guardados.
    """
    train_losses, train_accuracies, embeddings_history = [], [], []
    for epoch in range(epochs):
        model.train() # Modo entrenamiento
        optimizer.zero_grad()  # Clear existing gradients
        logits, embeddings = model(x, edge_index)
        # Calcular pérdida solo en nodos de entrenamiento
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        # Calcular precisión en entrenamiento
        pred = logits[train_mask].argmax(dim=1)
        acc = (pred == y[train_mask]).float().mean()
        # Guardar métricas
        train_losses.append(loss.item())
        train_accuracies.append(acc.item())
        # Guardar embeddings en épocas indicadas
        if epoch in save_epochs:
            embeddings_history.append(embeddings.detach().cpu().numpy())

    return train_losses, train_accuracies, embeddings_history