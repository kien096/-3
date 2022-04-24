from typing import Tuple

import torch
import torch.nn as nn

from config import NUMBER_CLASSES


class Model(nn.Module):
    def __init__(self, input_model):
        """
        :param input_model: input model from libraries.
        :param num_classes: number of classes.
        """
        super().__init__()
        self.input = input_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.input(x)
        output = self.softmax(x)

        return x, output
