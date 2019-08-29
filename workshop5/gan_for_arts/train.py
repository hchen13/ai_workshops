import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from data.loader import load_data, generate_labels
from prototypes import Generator, Discriminator
from utils import Monitor

image_size = 128
batch_size = 64
min_neurons = 64
noise_size = 128

pretrain_iterations = 2000
pretrain_learning_rate = 1e-4

train_iterations = 20000
gen_learning_rate = 1e-4
disc_learning_rate = 1e-4
critics = 3
gradient_penalty_weight = 1



def _create_generator_inputs(bs, num_classes):
    return generate_labels(bs, num_classes), tf.random.normal([bs, noise_size])

@tf.function
def generator_step(optimizer):
    with tf.GradientTape() as tape:
        generated_labels, noises = _create_generator_inputs(batch_size, num_classes)
        fake_images = gen.model([noises, generated_labels])
        fake_pred_logit, fake_pred_labels = disc.model(fake_images)
        gen_wasserstein_loss = -tf.reduce_mean(fake_pred_logit)
        gen_label_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(generated_labels, fake_pred_labels))
        gen_loss = gen_wasserstein_loss + gen_label_loss
    gradients = tape.gradient(gen_loss, gen.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gen.model.trainable_variables))
    return {
        'gen_wasserstein_loss': gen_wasserstein_loss,
        'gen_label_loss': gen_label_loss,
        'gen_loss': gen_loss,
    }


@tf.function
def discriminator_step(real_images, real_labels, optimizer):
    bs = len(real_images)
    with tf.GradientTape() as tape:
        generated_labels, noises = _create_generator_inputs(bs, num_classes)
        fake_images = gen.model([noises, generated_labels])
        real_logit, real_pred_labels = disc.model(real_images)
        fake_logit, fake_pred_labels = disc.model(fake_images)

        disc_w_loss = tf.reduce_mean(fake_logit) - tf.reduce_mean(real_logit)
        disc_label_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(real_labels, real_pred_labels)
        )
        alpha = tf.random.uniform([bs, 1, 1, 1])
        interpolates = (1 - alpha) * real_images + alpha * fake_images
        _logit, _ = disc.model(interpolates)
        grads = tf.gradients(_logit, interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=3))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1))
        disc_loss = disc_w_loss + disc_label_loss + gradient_penalty_weight * gradient_penalty
    gradients = tape.gradient(disc_loss, disc.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, disc.model.trainable_variables))
    return {
        'disc_wasserstein_loss': disc_w_loss,
        'disc_label_loss': disc_label_loss,
        'disc_gradient_penalty': gradient_penalty,
        'disc_loss': disc_loss,
    }


def eval(dataset, monitor, step):
    ys = []
    for i in range(len(dataset)):
        _, y = valid_gen[i]
        ys.append(y)
    y_true = np.concatenate(ys)
    _, y_pred = disc.model.predict(dataset)
    eq = np.equal(y_true.argmax(1), y_pred.argmax(1)).astype(np.float32)
    acc = np.mean(eq)
    monitor.scalar('valid_classification_acc', acc, step)

    labels = generate_labels(4, num_classes, condition=0)
    noise = tf.random.normal([4, noise_size])
    gen_images = gen.model.predict([noise, labels])
    monitor.image('generated', gen_images, step)


if __name__ == '__main__':
    print("\n[info] Loading image data...\n")
    data_root = os.path.join(os.path.expanduser('~'), 'datasets', 'artworks')
    train_gen, valid_gen = load_data(data_root, batch_size=batch_size, image_width=image_size, split=.05)
    num_classes = train_gen.num_classes

    print("\n[info] Creating generator and discriminator architectures for GAN...\n")
    gen = Generator(num_classes, image_size, bn=False)
    disc = Discriminator(num_classes, image_size, min_neurons)

    print("\n[info] Pre-training or loading pre-trained discriminator...\n")
    disc.pretrain(train_gen, valid_gen, pretrain_iterations, pretrain_learning_rate, retrain=False)

    ''' start training GAN for real '''
    print("\n[info] Start training GAN...\n")
    monitor = Monitor('GANGogh')
    gen_opt = tf.keras.optimizers.Adam(learning_rate=gen_learning_rate, beta_1=.5, beta_2=.9)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=disc_learning_rate, beta_1=.5, beta_2=.9)
    for i in range(train_iterations):
        print('iteration #{}'.format(i + 1))
        gen_losses = generator_step(gen_opt)
        for _ in range(critics):
            train_images, train_labels = next(train_gen)
            disc_losses = discriminator_step(train_images, train_labels, disc_opt)
        if i % 5 == 0:
            for tag, val in gen_losses.items():
                monitor.scalar(tag, val, int(i / 5))
            for tag, val in disc_losses.items():
                monitor.scalar(tag, val, int(i / 5))
        if (i < 1000 and i % 100 == 0) or (i % 1000 == 0) or i == train_iterations - 1:
            eval(valid_gen, monitor, i)
