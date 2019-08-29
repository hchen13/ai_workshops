import os
from glob import glob

import cv2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.utils import to_categorical


def norm_image(x):
    return x / 127.5 - 1


def denorm_image(x):
    return np.clip((x + 1) * 127.5, 0, 255).astype('uint8')


def load_data(path, batch_size, image_width, preprocess_func=norm_image, shuffle=True, split=0.):
    tdg = ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function=preprocess_func,
        validation_split=split
    )
    vdg = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        validation_split=split
    )
    train_gen = tdg.flow_from_directory(
        path,
        batch_size=batch_size,
        target_size=(image_width, image_width),
        shuffle=shuffle,
        subset='training'
    )
    valid_gen = vdg.flow_from_directory(
        path,
        batch_size=batch_size,
        target_size=(image_width, image_width),
        shuffle=shuffle,
        subset='validation'
    )
    return train_gen, valid_gen


def generate_labels(batch_size, num_classes, condition=None):
    if condition is None:
        labels = np.random.randint(num_classes, size=batch_size).astype(np.float32)
    else:
        labels = np.ones(shape=batch_size).astype(np.float32) * condition
    return to_categorical(labels, num_classes=num_classes)


if __name__ == '__main__':
    data_root = os.path.join(os.path.expanduser('~'), 'datasets', 'artworks')
    images = glob(os.path.join(data_root, '*/*.jpg'))
    for i, path in enumerate(images):
        img = cv2.imread(path)
        if img is None:
            print("Invalid image: {}".format(path), flush=True)
            os.remove(path)

