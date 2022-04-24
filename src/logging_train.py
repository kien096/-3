import os
import json
from typing import Dict, Tuple, List

from config import (BATCH_SIZE, LEARNING_RATE, BACKBONE, WEIGHTS, OUTPUT_ACTIVATION,
                    MODEL_NAME, AUGMENTATION_DATA, INPUT_SHAPE_IMAGE)


class LoggingModel:
    def __init__(self, save_path_log: str, list_augmentations, input_shape: Tuple = INPUT_SHAPE_IMAGE,
                 weights: str = WEIGHTS, model_name: str = MODEL_NAME, backbone: str = BACKBONE,
                 batch_size: int = BATCH_SIZE, learning_rate: float = LEARNING_RATE,
                 augmentation_data: bool = AUGMENTATION_DATA, output_activation: str = OUTPUT_ACTIVATION):
        """
        Logging_model is used to record the main parameters of the model that are used in training, as well as metrics.
        :param save_path_log: path for save json file with logs.
        :param list_augmentations: list of applied augmentation for data
        :param input_shape: input tensor, not considering batch size
        :param model_name: model name, for example "Unet"
        :param backbone: backbone name, for example "Resnet18"
        :param batch_size: input batch size
        :param learning_rate: input learning rate
        :param weights: pre-trained weights use or not, for example "imagenet weights"
        :param augmentation_data: use augmentation data or not
        :param output_activation: output activation, for example "softmax"
        :param layer_activation: layer_activation, for example "Relu"
        """

        aug_data = {augmentation_data: list_augmentations}
        self.log_file = os.path.join(save_path_log, 'train_logs.json')

        self.logs = {
            'Model_name': model_name,
            'Backbone': backbone,
            'Input_shape': input_shape,
            'Batch_size': batch_size,
            'Learning_rate': learning_rate,
            'Encoder_weights': weights,
            'Augmentation_data': aug_data,
            'Output_activation': output_activation,
        }

    def update_metric(self, metric: List, epoch: int) -> None:
        """
        :param metric: This is List object that contains a name and a value for the metric.
        """
        if metric[1] is Dict:
            if metric[0] not in self.logs:
                self.logs[metric[0]] = {}
                for name_class, value in metric[1].items():
                    self.logs[metric[0]][name_class] = []

            for name_class, value in metric[1].items():
                self.logs[metric[0]][name_class].append('epoch ' + str(epoch) + ': ' + str(metric[1]))

        else:
            if metric[0] not in self.logs:
                self.logs[metric[0]] = []
            self.logs[metric[0]].append('epoch ' + str(epoch) + ': ' + str(metric[1]))

    def save_json(self) -> None:
        """
        Save json file.
        """
        with open(self.log_file, 'w', encoding='utf-8') as file:
            json.dump(self.logs, file, indent=4, ensure_ascii=False)
