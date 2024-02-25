import torch
import warnings
from .classifiers import classifier, nn_classifier


class rewarder():
    def __init__(self) -> None:
        ...

    def reward(self, env: torch.Tensor) -> torch.Tensor:
        ...


class demo_rewarder(rewarder):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def reward(env: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(env) == 0:
            warnings.warn("The length of env must > 0!")
            env = torch.rand(10, 2)

        output = torch.softmax(env, dim=env.dim()-1)
        output = torch.square(output)
        output = torch.sum(output, dim=env.dim()-1)
        output = torch.reshape(output, (-1, 1))
        return output


class classify_rewarder(rewarder):
    def __init__(self, clf: classifier) -> None:
        super().__init__()
        self.clf = clf
        self.reward_item = 0

    def reward(self, env: torch.Tensor) -> torch.Tensor:
        """ dim of env < 3

        Args:
            env (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        if len(env) == 0:
            raise RuntimeError("The length of env must > 0!")

        classification = self.clf.classify(env)
        classification_dim = classification.dim()

        if classification_dim > 2:
            raise RuntimeError("classification_dim must < 3 !")

        output = classification[(slice(None),) *
                                classification_dim + (self.reward_item,)]
        output = torch.reshape(output, (-1, 1))

        return output

    def set_item_class(self, item_idx: int) -> None:
        self.reward_item = item_idx
