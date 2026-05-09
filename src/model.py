"""Model factory — routes to FC or CNN based on params."""

from src.models.fc import SensorClassifier
from src.models.cnn import SensorCNN1D

MODEL_REGISTRY = {
    "fc": SensorClassifier,
    "cnn": SensorCNN1D,
}


def get_model(arch: str, **kwargs):
    """Return model instance by architecture name.

    Args:
        arch: Architecture identifier ('fc' or 'cnn').
        **kwargs: Arguments passed to the model constructor.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If arch is not in MODEL_REGISTRY.
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](**kwargs)