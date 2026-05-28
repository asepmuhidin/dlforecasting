import torch
import torch.nn as nn

from tqdm.auto import tqdm

from .callbacks import EarlyStopping


def train_model(
    model,
    train_loader,
    epochs=50,
    lr=0.001,
    patience=10,
    device='cpu',
    verbose=True
):

    model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    early_stopping = EarlyStopping(
        patience=patience
    )

    history = []

    if verbose:
        pbar = tqdm(range(epochs))
    else:
        pbar = range(epochs)

    for epoch in pbar:

        model.train()

        total_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            output = model(X_batch)

            loss = criterion(
                output,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        history.append(avg_loss)

        if verbose:

            pbar.set_description(
                f"Epoch {epoch+1}/{epochs}"
            )

            pbar.set_postfix(
                avg_loss=f"{avg_loss:.6f}"
            )

        early_stopping(avg_loss)

        if early_stopping.early_stop:

            print(
                "Early stopping triggered"
            )

            break

    return history