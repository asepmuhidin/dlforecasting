import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import torch


def evaluate(
    model,
    loader,
    scaler,
    device='cpu'
):

    model.eval()

    preds = []

    actuals = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            output = model(X_batch)

            preds.extend(
                output.cpu().numpy()
            )

            actuals.extend(
                y_batch.numpy()
            )

    preds = np.array(preds)

    actuals = np.array(actuals)

    preds_actual = scaler.inverse_transform(
        preds.reshape(-1, 1)
    ).flatten()

    actuals_actual = scaler.inverse_transform(
        actuals.reshape(-1, 1)
    ).flatten()

    mse = mean_squared_error(
        actuals_actual,
        preds_actual
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        actuals_actual,
        preds_actual
    )

    r2 = r2_score(
        actuals_actual,
        preds_actual
    )

    mape = np.mean(
        np.abs(
            (
                actuals_actual
                - preds_actual
            )
            /
            (
                actuals_actual
                + 1e-8
            )
        )
    ) * 100

    return {

        'mse': mse,

        'rmse': rmse,

        'mae': mae,

        'mape': mape,

        'r2': r2,

        'predictions': preds_actual,

        'actuals': actuals_actual
    }