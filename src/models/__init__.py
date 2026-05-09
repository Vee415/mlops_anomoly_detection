from src.models.fc import SensorClassifier
from src.models.cnn import SensorCNN1D

MODEL_REGISTRY = {
    "fc": SensorClassifier,
    "cnn": SensorCNN1D,
}