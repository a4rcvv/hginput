import torchinfo
from hginput.model.model import GestureClassifier


def summary():
    model = GestureClassifier(n_classes=2)
    torchinfo.summary(model, input_size=(32, 64))
