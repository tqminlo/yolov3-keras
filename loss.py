import tensorflow as tf
from keras.layers import *
from keras.models import Model


def yolov3_loss(y_true, y_pred):
    '''
    Calculate loss from y_true and y_pred which are list of grid-map-predict
    Example about yolov3 default has 3 grid-map-predict:
    :param y_true: [{shape = (batch, 13, 13, 3, numclasses+8)}, {shape = (batch, 26, 26, 3, numclasses+8)}, {shape = (batch, 52, 52, 3, numclasses+8)}]
    :param y_pred: [{shape = (batch, 13, 13, 3, numclasses+8)}, {shape = (batch, 26, 26, 3, numclasses+8)}, {shape = (batch, 52, 52, 3, numclasses+8)}]
    :return: loss, shape = (1,)
    '''
    # print(type(y_true))
    # print(type(y_pred))
    # Demo with mean-squared-error loss, not yolo loss
    loss = tf.reduce_mean((y_true[0] - y_pred[0]) ** 2) + tf.reduce_mean((y_true[1] - y_pred[1]) ** 2) + tf.reduce_mean((y_true[2] - y_pred[2]) ** 2)
    loss = loss/3
    return loss

