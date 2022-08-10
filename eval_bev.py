import pickle
import numpy as np


def evaluate_map(gt_data, pred_data):
    thresholds = np.array([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    map_classes = ['drivable_area', 'divider']
    num_classes = len(map_classes)
    num_thresholds = len(thresholds)

    tp = np.zeros((num_classes, num_thresholds))
    fp = np.zeros((num_classes, num_thresholds))
    fn = np.zeros((num_classes, num_thresholds))

    for i, (pred, label) in enumerate(zip(pred_data, gt_data)):
        pred = pred.reshape(num_classes, -1)
        label = label.reshape(num_classes, -1)

        pred = np.where(pred[:, :, None] >= thresholds, 1, 0)
        label = label[:, :, None]

        tp += np.multiply(pred, label).sum(axis=1)
        fp += np.multiply(pred, 1 - label).sum(axis=1)
        fn += np.multiply(1 - pred, label).sum(axis=1)

    ious = tp / (tp + fp + fn + 1e-7)

    metrics = {}
    for index, name in enumerate(map_classes):
        metrics[f"map/{name}/iou@max"] = ious[index].max().item()
        for threshold, iou in zip(thresholds, ious[index]):
            metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
    metrics["map/mean/iou@max"] = ious.max(axis=1).mean().item()
    return metrics
