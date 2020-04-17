from functools import partial
import tensorflow
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """ 
        Dice coefficent. Works for 2D and 3D Labels.
        Dice coefficent = (2 * |A ∩ B|) / (|A ⋃ B|)
                = sum(|A * B|) / (sum(|A|) + sum(|B|))
        https://arxiv.org/pdf/1707.03237.pdf
        Makes no assumptions about the rank of y_true and y_pred
        :param y_true
        :param y_pred
        :param smooth
        :return scalar
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def weighted_dice_coefficient(y_true, y_pred, axis=(-4, -3, -2), smooth=1e-6):
    """ 
        Weighted dice coefficient.
        Assumes y_true and y_pred are of rank 5.
        :param smooth:      default = 1e-5
        :param y_true:
        :param y_pred:
        :param axis:        default assumes a "channels-last" data structure
        :return: scalar
    """
    return K.mean(
        2.0
        * (K.sum(y_true * y_pred, axis=axis) + smooth / 2)
        / (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth)
    )


def volume_aware_dice_coefficent(y_true, y_pred):
    """

    """
    print("TODOOOOOO")


def per_volume_dice_coefficient(y_true, y_pred):
    """
        Calculates and returns the dice coefficent label wise and returns a list of the score and label index for each label volume.
        Assumes tensors of rank 5 for y_true and y_pred. 
        :param y_true
        :param y_pred
    """
    dsc_scores = []
    for l in range(y_true.shape[0]):
        dsc = dice_coefficient(y_true[l], y_pred[l])
        dsc_scores.append(l, dsc)

    return dsc_scores


dice_coefficient = dice_coefficient
weighted_dice_coefficient = weighted_dice_coefficient
