import numpy as np


def mean_IoU(predicted, target):
    iousum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone(
        ).detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone(
        ).detach().cpu().numpy().argmax(0)

        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else:
            iou_score = intersection / union
        iousum += iou_score

    miou = iousum/target.shape[0]
    return miou


def pixel_accuracy(predicted, target):
    accuracy_sum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone(
        ).detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone(
        ).detach().cpu().numpy().argmax(0)

        same = (target_arr == predicted_arr).sum()
        a, b = target_arr.shape
        total = a*b
        accuracy_sum += same/total

    pixelAccuracy = accuracy_sum/target.shape[0]
    return pixelAccuracy
