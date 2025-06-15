import torch
import torch.nn.functional as F

def train(model, x, edge_index, y, train_mask, optimizer, criterion, epochs, save_epochs):
    train_losses, train_accuracies, embeddings_history = [], [], []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, embeddings = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        pred = logits[train_mask].argmax(dim=1)
        acc = (pred == y[train_mask]).float().mean()

        train_losses.append(loss.item())
        train_accuracies.append(acc.item())

        if epoch in save_epochs:
            embeddings_history.append(embeddings.detach().cpu().numpy())

    return train_losses, train_accuracies, embeddings_history