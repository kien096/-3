import os
import timm
from typing import Dict, Optional, Union, Tuple
from shutil import copy2
from itertools import islice

import torch

from src.model import Model


def save_checkpoint(state: Dict, is_best: bool, path_state: str, best_state_path: str, name_model: str) -> None:
    """
    :param name_model:
    :param state: dictionary that includes model parameters to save.
    :param is_best: is this state of the model better than the previous one or not.
    :param path_state: path to save the current state of the model.
    :param best_state_path: path to save the best state of the model.
    """
    torch.save(state, os.path.join(path_state, 'checkpoint.pth'))
    if is_best:
        copy2(os.path.join(path_state, 'checkpoint.pth'), os.path.join(best_state_path, name_model))


def load_check_point(checkpoint_path: str, model, optimizer, device: torch.device):
    """
    checkpoint_path: path to save checkpoint file
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer that we want to load state
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def load_model_weigts(model_name: str, num_classes: int, pretrained: bool, in_chans: int, device: torch.device,
                      weights: Optional[str] = None):
    """
    This function builds a model, you can also load trained weights.
    :param weights: path to .pth file.
    :param model_name: name of the model to build.
    :param num_classes: number of classes.
    :param pretrained: load imagnet weights or not.
    :param in_chans: conversion of input channels to three channels, for training on imagenet scales.
    :param device: place where input data is loaded.
    :return: model
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
    model = Model(model)

    if weights is not None:
        load_parameters = torch.load(weights, map_location=torch.device(device))
        # sliced = load_parameters["state_dict"]
        # sliced.pop("input.classifier.bias")
        # sliced.pop("input.classifier.weight")
        model.load_state_dict(load_parameters['state_dict'], strict=False)

    model.to(device)

    return model


def prepare_results_metrics_print(mean_metrics: Dict, metrics_each_class: Optional[Dict] = None,
                                  json_class: Dict = None) -> Union[Tuple, str]:
    """
    This function makes print metrics more accurate.
    :param mean_metrics: mean value for each metric.
    :param metrics_each_class: mean value for each class according to a certain metric.
    :param json_class: name of classes with id.
    :return: string for print.
    """
    metrics = ''

    for key, value in mean_metrics.items():
        metrics += key + ': ' + '{:4f}'.format(value.item()) + '\n'

    if metrics_each_class is not None:
        metrics_for_each_class = '\n\n'
        for key, value in metrics_each_class.items():
            metrics_for_each_class += '\n' + key + '  _____________________________________\n'
            for i, val_tensor in enumerate(value):
                name = list(json_class.keys())[list(json_class.values()).index(i)]
                metrics_for_each_class += name + ': ' + '{:4f}'.format(val_tensor.item()) + '\n'

        return metrics, metrics_for_each_class

    else:
        return metrics


from typing import Union

import numpy as np
import soundfile as sf
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from librosa.display import specshow


def plot_signal(first, second=None):

    figure, axis = plt.subplots(3)
    axis[0].plot(first)
    axis[1].plot(second)

    plt.show()
    if plt.waitforbuttonpress(0):
        plt.close('all')
        return
    plt.close('all')


def write_sound(wav: Union[np.memmap, np.array], sr: int, name: str):
    sf.write(name, wav, sr, format='wav')


def show_spectogramm(data_1):
    spec_1, sr_1 = data_1
    plt.figure()
    specshow(spec_1, sr=sr_1, x_axis='time', y_axis='mel', cmap=cm.magma)
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    if plt.waitforbuttonpress(0):
        plt.close('all')
        return
    plt.close('all')
