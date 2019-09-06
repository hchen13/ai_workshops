import numpy as np
import tensorflow as tf
from PIL import Image

from data.loader import generate_labels, denorm_image
from prototypes import Generator, Discriminator

idx2label = {
    0: 'abstract',
    1: 'animal-painting',
    2: 'cityscape',
    3: 'figurative',
    4: 'flower-painting',
    5: 'genre-painting',
    6: 'landscape',
    7: 'marina',
    8: 'mythological-painting',
    9: 'nude-painting-nu',
    10: 'portrait',
    11: 'religious-painting',
    12: 'still-life',
    13: 'symbolic-painting'
}
label2idx = {val: key for key, val in idx2label.items()}
num_classes = len(idx2label.keys())
noise_size = 128
image_size = 128

gen = Generator(num_classes, image_size, bn=False)
disc = Discriminator(num_classes, image_size, 64, bn_epsilon=0)
gen.load('GANGogh')
disc.load('GANGogh')

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


class Candidate:
    def __init__(self, arr_image, critic_score, label, label_confidence):
        self.image = arr_image
        self.critic_score = critic_score
        self.label = label
        self.label_confidence = label_confidence


def select_best_images(label, num_samples):
    LOOK_AT = 1
    BATCH_SIZE = 64
    input_label = generate_labels(BATCH_SIZE, num_classes, condition=label)
    list_candidates = []
    for j in range(LOOK_AT):
        noise = tf.random.uniform(shape=[BATCH_SIZE, noise_size], minval=-1., maxval=1.)
        samples = gen.model.predict([noise, input_label])
        pred_realness, pred_labels = disc.model.predict(samples)
        pred_realness = pred_realness.squeeze()
        guess = np.argmax(pred_labels, axis=1)
        confidence = np.amax(pred_labels, axis=1)
        indices = list(np.argwhere(guess == label))
        samples = denorm_image(samples)
        for k in indices:
            k = k.squeeze()
            candidate = Candidate(samples[k], pred_realness[k], label, confidence[k])
            list_candidates.append(candidate)
    list_candidates.sort(key=lambda x: x.label_confidence, reverse=True)
    list_candidates = list_candidates[:num_samples * 3]
    list_candidates.sort(key=lambda x: x.critic_score, reverse=True)
    list_candidates = list_candidates[:num_samples]
    return list_candidates


def generate_images(num_samples):
    noise = tf.random.normal(shape=[num_samples, noise_size])
    labels = generate_labels(num_samples, num_classes)
    images = gen.model.predict([noise, labels])
    return images


def create_montage(images, cols, width=1000):
    _width = int(width / cols)
    rows = np.math.ceil(len(images) / cols)
    height = np.ceil(_width * rows).astype('int')
    canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
    for row in range(rows):
        for col in range(cols):
            arr = images[row * cols + col]
            x0, y0 = col * _width, row * _width
            x1, y1 = x0 + _width, y0 + _width
            image = Image.fromarray(arr)
            resized = image.resize((_width, _width))
            canvas.paste(resized, box=(x0, y0, x1, y1))
    return canvas


if __name__ == '__main__':
    images = denorm_image(generate_images(24))
    montage = create_montage(images, cols=6, width=750)
    display_image(np.array(montage), width=9)
    # montage.save('random_arts.jpg')

    # genre = 'portrait'
    # candidates = select_best_images(6, label2idx[genre])
    # images = [i.image for i in candidates]
    # display_image(*images, col=6, width=15)
    # montage = create_montage(images, cols=2, width=750)
    # montage.save('selected_{}.jpg'.format(genre))