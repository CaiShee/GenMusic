import torch
from torch import nn


class classifier():
    def __init__(self) -> None:
        ...

    def classify(self, feature: torch.Tensor) -> torch.Tensor:
        ...


class nn_classifier(classifier):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def classify(self, feature: torch.Tensor) -> torch.Tensor:
        return self.model.forward(feature)

def get_from_pretrained_nn(md_path: str) -> nn_classifier:
    model = torch.load(md_path)
    return nn_classifier(model)