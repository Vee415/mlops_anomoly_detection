from src.models.cnn import SensorCNN1D
from src.models.fc import SensorClassifier

MODEL_REGISTRY = {
    "fc": SensorClassifier,
    "cnn": SensorCNN1D,
}
