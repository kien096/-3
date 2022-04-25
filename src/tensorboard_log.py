from typing import Dict

from config import NUMBER_CLASSES


def number(num, digits=4) -> float:
    return float(f'{num:.{digits}f}')


def log_tensorboard(tensorboard, train_metric, val_metric, num_epoch, train_metric_each_class=None, log=None,
                    val_metric_each_class=None, epoch_loss=None, val_epoch_loss=None,
                    log_echa_metrics: bool = False, save_logs_class_nun_epoch: int = 10) -> Dict:
    """
    :param tensorboard: tensorboard object for entering information.
    :param train_metric: averaged trainig metric by epoch.
    :param val_metric: avarage validation metric by epoch.
    :param epoch_loss: avarage training loss by epoch.
    :param val_epoch_loss: avarage validation loss by epoch.
    :param num_epoch: epoch number.
    :param train_metric_each_class: avarage training metrics for each class.
    :param val_metric_each_class: avaraged validation metrics for each class.
    :param class_json: dict with indexes for each class.
    :param log: object for logging data.
    :param log_echa_metrics:
    :param save_logs_class_nun_epoch:
    :return dictionary with structured information about losses and metrics.
    """

    if epoch_loss is not None or val_epoch_loss is not None:
        tensorboard.add_scalars('Loss', {"train": epoch_loss, "val": val_epoch_loss}, num_epoch)
        log.update_metric(['train_loss', number(epoch_loss)], num_epoch)
        log.update_metric(['val_loss', number(val_epoch_loss)], num_epoch)

    dict_with_data = {}
    print_metric = {'Loss': {"train": number(epoch_loss), "val": number(val_epoch_loss)}}

    for all_metrcis in [train_metric.items(), val_metric.items()]:
        for name_metric, value_tensor in all_metrcis:
            name_draw = name_metric.split('_')[-1]
            val_or_train = name_metric.split('_')[0]
            if val_or_train == 'train':
                dict_with_data[val_or_train] = value_tensor.item()
            if val_or_train == 'val':
                dict_with_data[val_or_train] = value_tensor.item()

            log.update_metric([val_or_train + '_' + name_draw, value_tensor.item()], num_epoch)
            print_metric[val_or_train + '_' + name_draw] = number(value_tensor.item())
            tensorboard.add_scalars(name_draw, dict_with_data, num_epoch)

    log.save_json()

    return print_metric