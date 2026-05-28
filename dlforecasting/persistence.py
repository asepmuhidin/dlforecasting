import torch
import joblib

from .models import HybridModel


def save_model(
    model,
    config,
    metrics,
    path='model.pth'
):
    """
    Save trained model.
    """

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics
        },
        path
    )


def load_model(path):
    """
    Load trained model.
    """

    checkpoint = torch.load(path)

    config = checkpoint['config']

    model = HybridModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        model_type=config['model_type']
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    model.eval()

    return model, checkpoint


def save_scaler(
    scaler,
    path='scaler.save'
):
    """
    Save scaler object.
    """

    joblib.dump(
        scaler,
        path
    )


def load_scaler(
    path='scaler.save'
):
    """
    Load scaler object.
    """

    return joblib.load(path)