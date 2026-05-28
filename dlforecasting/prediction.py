import numpy as np
import torch


def predict_future(
    model,
    last_sequence,
    n_future=10,
    device='cpu'
):
    """
    Predict future values using trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    last_sequence : ndarray
        Last input sequence.

    n_future : int
        Number of future steps.

    device : str
        cpu or cuda

    Returns
    -------
    ndarray
        Predicted values.
    """

    model.eval()

    predictions = []

    current_seq = last_sequence.copy()

    with torch.no_grad():

        for _ in range(n_future):

            x = torch.tensor(
                current_seq,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            pred = model(x).item()

            predictions.append(pred)

            pred_array = np.array([[pred]])

            current_seq = np.vstack([
                current_seq[1:],
                pred_array
            ])

    return np.array(predictions)