import torch

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader

from .utils import (
    set_seed,
    create_sequences,
    get_device
)

from .datasets import TimeSeriesDataset

from .models import HybridModel

from .training import train_model

from .evaluation import evaluate

from .persistence import (
    save_model,
    save_scaler
)


def dlforecasting(

    series,

    train_size=0.8,

    input_size=1,

    hidden_size=64,

    num_layers=1,

    model_type='LSTM',

    epoch_size=50,

    window_size=30,

    batch_size=32,

    lr=0.001,

    patience=10,

    seed=42,

    save=True,

    model_path=None,

    scaler_path=None,

    device=None,

    verbose=True
):
    """
    Deep Learning Time Series Forecasting.

    Parameters
    ----------
    series : pandas.DataFrame
        Time series dataset.

    train_size : float
        Training data ratio.

    input_size : int
        Number of input features.

    hidden_size : int
        Hidden layer size.

    num_layers : int
        Number of recurrent layers.

    model_type : str
        RNN, LSTM, or GRU.

    epoch_size : int
        Number of epochs.

    window_size : int
        Sequence length.

    batch_size : int
        Batch size.

    lr : float
        Learning rate.

    patience : int
        Early stopping patience.

    seed : int
        Random seed.

    save : bool
        Save model or not.

    device : str
        cpu or cuda.

    verbose : bool
        Show progress bar.

    Returns
    -------
    dict
        Training result.
    """

    # =====================================================
    # RANDOM SEED
    # =====================================================
    set_seed(seed)

    # =====================================================
    # DEVICE
    # =====================================================
    if device is None:

        device = get_device()

    # =====================================================
    # SPLIT DATA
    # =====================================================
    split_idx = int(
        len(series) * train_size
    )

    train = series.iloc[:split_idx]

    test = series.iloc[split_idx:]

    # =====================================================
    # SCALING
    # =====================================================
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train
    )

    test_scaled = scaler.transform(
        test
    )

    # =====================================================
    # WINDOWING
    # =====================================================
    X_train, y_train = create_sequences(
        train_scaled,
        window_size
    )

    X_test, y_test = create_sequences(
        test_scaled,
        window_size
    )

    # =====================================================
    # DATASET
    # =====================================================
    train_dataset = TimeSeriesDataset(
        X_train,
        y_train
    )

    test_dataset = TimeSeriesDataset(
        X_test,
        y_test
    )

    # =====================================================
    # DATALOADER
    # =====================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # =====================================================
    # MODEL
    # =====================================================
    model = HybridModel(

        input_size=input_size,

        hidden_size=hidden_size,

        num_layers=num_layers,

        model_type=model_type
    )

    model.to(device)

    # =====================================================
    # TRAINING
    # =====================================================
    history = train_model(

        model=model,

        train_loader=train_loader,

        epochs=epoch_size,

        lr=lr,

        patience=patience,

        device=device,

        verbose=verbose
    )

    # =====================================================
    # EVALUATION
    # =====================================================
    metrics = evaluate(

        model=model,

        loader=test_loader,

        scaler=scaler,

        device=device
    )

    # =====================================================
    # CONFIG
    # =====================================================
    config = {

        'train_size': train_size,

        'input_size': input_size,

        'hidden_size': hidden_size,

        'num_layers': num_layers,

        'model_type': model_type,

        'epoch_size': epoch_size,

        'window_size': window_size,

        'batch_size': batch_size,

        'learning_rate': lr,

        'patience': patience,

        'seed': seed
    }

    # =====================================================
    # SAVE MODEL
    # =====================================================
    if save:

        if model_path is None:

            model_path = (
                f"{model_type.lower()}_"
                f"window{window_size}_"
                f"hidden{hidden_size}.pth"
            )

        save_model(

            model=model,

            config=config,

            metrics={

                'mse': metrics['mse'],

                'rmse': metrics['rmse'],

                'mae': metrics['mae'],

                'mape': metrics['mape'],

                'r2': metrics['r2']
            },

            path=model_path
        )

        # ================================================
        # SAVE SCALER
        # ================================================
        if scaler_path is None:

            scaler_path = 'scaler.save'

        save_scaler(

            scaler=scaler,

            path=scaler_path
        )

    # =====================================================
    # RETURN
    # =====================================================
    return {

        'model': model,

        'history': history,

        'metrics': metrics,

        'config': config,

        'scaler': scaler,

        'device': device
    }