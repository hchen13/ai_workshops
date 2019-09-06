import os
import shutil

import numpy as np
import tensorflow as tf

class Monitor:
    def __init__(self, caption):
        log_root = os.path.join(os.path.dirname(__file__), '.logs')
        fullpath = os.path.join(log_root, caption)
        try:
            shutil.rmtree(fullpath)
        except FileNotFoundError:
            pass
        os.makedirs(fullpath, exist_ok=True)
        self.logdir = fullpath
        self.caption = caption
        self.writer = tf.summary.create_file_writer(fullpath)

    def scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, data=value, step=step)

    def image(self, tag, images, step):
        with self.writer.as_default():
            tf.summary.image(tag, images, max_outputs=64, step=step)


def display_image(*images, col=None, width=20):
    from matplotlib import pyplot as plt
    if col is None:
        col = len(images)
    row = np.math.ceil(len(images) / col)
    plt.figure(figsize=(width, (width + 1) * row / col))
    for i, image in enumerate(images):
        plt.subplot(row, col, i + 1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()