import torch
from torch import nn
from torchmetrics import Accuracy, Precision, Recall, MetricCollection, F1Score

from config import NUMBER_CLASSES


class MetricsTorch(nn.Module):
    def __init__(self, average: str, device: torch.device, num_classes: int = NUMBER_CLASSES, accuracy: bool = False,
                 precision: bool = False, recall: bool = False, f1: bool = False, each_class: bool = False):
        """
        This class is used to calculate metrics
        :param average: Defines the reduction that is applied. Should be one of the following ["micro", "macro",
        "weighted", "samples", "none", None].
        :param device: where the metrics will be: 'cpu', 'cuda' or something else.
        :param num_classes: number of classes.
        :param accuracy: calculate 'accuracy' or not.
        :param precision: calculate 'precision' or not.
        :param recall: calculate 'recall' or not.
        :param f1: calculate 'f1' or not.
        :param each_class: calculate metrics for each class ot not.
        """
        super(MetricsTorch, self).__init__()

        collection_metcics = []

        if accuracy:
            accuracy_ = Accuracy(average=average, num_classes=num_classes)
            collection_metcics.append(accuracy_)
        if precision:
            precision_ = Precision(average=average, num_classes=num_classes)
            collection_metcics.append(precision_)
        if recall:
            recall_ = Recall(average=average, num_classes=num_classes)
            collection_metcics.append(recall_)
        if f1:
            f1_ = F1Score(average=average, num_classes=num_classes)
            collection_metcics.append(f1_)

        if each_class:
            collection_metcics_each_class = []
            if accuracy:
                accuracy_ = Accuracy(average='none', num_classes=num_classes)
                collection_metcics_each_class.append(accuracy_)
            if precision:
                precision_ = Precision(average='none', num_classes=num_classes)
                collection_metcics_each_class.append(precision_)
            if recall:
                recall_ = Recall(average='none', num_classes=num_classes)
                collection_metcics_each_class.append(recall_)
            if f1:
                f1_ = F1Score(average='none', num_classes=num_classes)
                collection_metcics_each_class.append(f1_)

        metrics = MetricCollection(collection_metcics).to(device=device)

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

        metrcis_each_class = MetricCollection(collection_metcics_each_class).to(device=device)

        self.train_metrics_each_class = metrcis_each_class.clone(prefix='train_')
        self.valid_metrics_each_class = metrcis_each_class.clone(prefix='val_')

    def training_step(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        """
        Calculation of metrics for each training batch.
        :param labels: true values.
        :param predictions: predicted values.
        """
        self.train_metrics.update(labels, predictions)

    def validation_step(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        """
        Calculation of metrics for each validation batch.
        :param labels: true values.
        :param predictions: predicted values.
        """
        self.valid_metrics.update(labels, predictions)

    def training_step_each_class(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        """
        Calculation of metrics for each trainig classes.
        :param labels: true values.
        :param predictions: predicted values.
        """
        self.train_metrics_each_class.update(labels, predictions)

    def validation_step_each_class(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        """
        Calculation of metrics for each validation classes.
        :param labels: true values.
        :param predictions: predicted values.
        """
        self.valid_metrics_each_class.update(labels, predictions)

    def training_step_compute(self):
        """
        Calculation of the average training metrics.
        """
        return self.train_metrics.compute()

    def validation_step_compute(self):
        """
        Calculation of the average validation metrics.
        """
        return self.valid_metrics.compute()

    def training_step_each_class_compute(self):
        """
        Calculation of the average training metrics for each class.
        """
        return self.train_metrics_each_class.compute()

    def validation_step_each_class_compute(self):
        """
        Calculation of the average validation metrics for each class.
        """
        return self.valid_metrics_each_class.compute()

    def training_step_reset(self) -> None:
        """
        Zeroing the metrics at the end of an epoch.
        """
        self.train_metrics.reset()

    def validation_step_reset(self) -> None:
        """
        Zeroing the metrics at the end of an epoch.
        """
        self.valid_metrics.reset()

    def training_step_each_class_reset(self) -> None:
        """
        Zeroing the metrics at the end of an epoch.
        """
        self.train_metrics_each_class.reset()

    def validation_step_each_class_reset(self) -> None:
        """
        Zeroing the metrics at the end of an epoch.
        """
        self.valid_metrics_each_class.reset()
