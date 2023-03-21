import numpy as np
import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from yolov3 import YOLOv3
from loss import yolov3_loss



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def data_generator_wrapper(annotation_lines, batch_size, input_shape, num_classes):
    return data_generator(annotation_lines, batch_size, input_shape, num_classes)


def data_generator(annotation_lines, batch_size, input_shape, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        images_train = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image_train, box = get_data_from_annotation(annotation_lines[i], input_shape)
            images_train.append(image_train)
            box_data.append(box)
            i = (i+1) % n
        images_train = np.array(images_train)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, num_classes)
        yield images_train, y_true
        # return images_train, y_true


def get_data_from_annotation(annotation_line, input_shape, max_boxes=3):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image_path = line[0]
    image_arr = cv2.imread(image_path)
    iw, ih = image_arr.shape[0], image_arr.shape[1]
    h, w = input_shape
    box = np.array([[int(value) for value in (box.split(','))] for box in line[1:]])

    # reshape image
    image_data = cv2.resize(image_arr, (w, h))/255.

    # correct boxes
    scale_w = w/iw
    scale_h = h/ih
    box_data = np.zeros((max_boxes, 5))
    box_data[:, 4] = -1
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes: box = box[:max_boxes]
        box[:, [0,2]] = box[:, [0,2]] * scale_w
        box[:, [1,3]] = box[:, [1,3]] * scale_h
        box_data[:len(box)] = box

    return image_data, box_data


def preprocess_true_boxes(box_data, input_shape, num_classes):
    # assert (box_data[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_grid_map = 3
    num_box_per_cell = 1
    max_box_per_img = box_data.shape[1]
    box_data = np.array(box_data, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (box_data[..., 0:2] + box_data[..., 2:4]) // 2
    boxes_wh = box_data[..., 2:4] - box_data[..., 0:2]
    box_data[..., 0:2] = boxes_xy / input_shape
    box_data[..., 2:4] = boxes_wh / input_shape

    batch_size = box_data.shape[0]
    if num_grid_map > 1:
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_grid_map)]
    else:
        grid_shapes = [input_shape // {0: 16}[l] for l in range(num_grid_map)]  # num_layers = 1, grid_shapes = (26, 26)

    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], num_box_per_cell, 5+num_classes),
        dtype='float32') for l in range(num_grid_map)]
    for l in range(num_grid_map):
        for n in range(batch_size):
            for b in range(max_box_per_img):
                c = box_data[n, b, 4].astype('int32')
                if c > 0:
                    i = np.floor(box_data[n, b, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(box_data[n, b, 1] * grid_shapes[l][0]).astype('int32')
                    # k = anchor_mask[l].index(n)
                    c = box_data[n, b, 4].astype('int32')
                    y_true[l][n, j, i, 0, 0:4] = box_data[n, b, 0:4]
                    y_true[l][n, j, i, 0, 4] = 1
                    y_true[l][n, j, i, 0, 5+c] = 1
    return y_true
    # y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], 5+num_classes),
    #     dtype='float32') for l in range(num_grid_map)]
    # for l in range(num_grid_map):
    #     for n in range(batch_size):
    #         for b in range(max_box_per_img):
    #             c = box_data[n, b, 4].astype('int32')
    #             if c > 0:
    #                 i = np.floor(box_data[n, b, 0] * grid_shapes[l][1]).astype('int32')
    #                 j = np.floor(box_data[n, b, 1] * grid_shapes[l][0]).astype('int32')
    #                 # k = anchor_mask[l].index(n)
    #                 c = box_data[n, b, 4].astype('int32')
    #                 y_true[l][n, j, i, 0:4] = box_data[n, b, 0:4]
    #                 y_true[l][n, j, i, 4] = 1
    #                 y_true[l][n, j, i, 5+c] = 1
    # return y_true[0]


def train(size, epochs, batch_size, pre_trained=False, pre_model_path=None):
    # get some cfg params
    annotation_path = f'dataset_{size}/labels.txt'
    classes_path = 'config/ceatec_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    input_shape = (size, size)

    # get num train/val
    val_split = 0.25
    with open(annotation_path) as f:
        lines = f.readlines()
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # create model
    if pre_trained and pre_model_path:
        model = tf.keras.models.load_model(pre_model_path)
    else:
        model = YOLOv3(size, num_classes).model()
    model.summary()
    model.compile(optimizer=Adam(lr=1e-3), loss=yolov3_loss)

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=35),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=10)]

    model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(lines[num_train:], batch_size, input_shape, num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=epochs,
                        callbacks=my_callbacks)

    model.save(f"saved_models/yolov3_{size}.h5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument('-tm', '--train_mode', help='train mode', default="tinyv3-custom")
    ap.add_argument('-s', '--size', help='train mode', type=int, default=416)
    args = ap.parse_args()

    # print("---TRAIN MODE---: ", args.train_mode)
    print("---TRAIN SIZE___: ", args.size)

    train(size=args.size, epochs=300, batch_size=8)