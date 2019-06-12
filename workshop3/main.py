from datetime import datetime

from IPython import display
from tensorflow.contrib import eager
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.vgg19 import preprocess_input, VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


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
    plt.show()


def preprocess(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)


def deprocess(image):
    x = image.copy()
    x = np.squeeze(x, 0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.680
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():
    vgg = VGG19(include_top=False)
    vgg.trainable = False
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    return Model(vgg.input, style_outputs + content_outputs)


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(x):
    channels = int(x.shape[-1])
    a = tf.reshape(x, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(generated_style, base_gram_style):
    gram_style = gram_matrix(generated_style)
    return tf.reduce_mean(tf.square(gram_style - base_gram_style)) * .75


def get_feature_repr(model, content_image, style_image):
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    style_features = [layer[0] for layer in style_outputs[:num_style_layers]]
    content_features = [layer[0] for layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_loss(model, generated_image, gram_style_features, content_features, content_weight, style_weight):
    model_outputs = model(generated_image)
    output_style_features = model_outputs[:num_style_layers]
    output_content_features = model_outputs[num_style_layers:]

    content_loss, style_loss = 0, 0

    w = 1. / num_style_layers
    for target_style, gen_style in zip(gram_style_features, output_style_features):
        style_loss += w * get_style_loss(gen_style[0], target_style)

    w = 1. / num_content_layers
    for target_content, gen_content in zip(content_features, output_content_features):
        content_loss += w * get_content_loss(gen_content[0], target_content)

    content_loss *= content_weight
    style_loss *= style_weight

    loss = content_loss + style_loss
    return loss, style_loss, content_loss


def compute_grads(configs):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**configs)
    loss = all_loss[0]
    return tape.gradient(loss, configs['generated_image']), all_loss


def run(content_path, style_path, num_iters=1000, content_weight=1e3, style_weight=1e-2, lr=10):
    content_image = preprocess(content_path)
    style_image = preprocess(style_path)

    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_repr(model, content_image, style_image)
    gram_style_features = [gram_matrix(x) for x in style_features]

    init_image = preprocess(content_path)
    generated_image = eager.Variable(init_image, dtype=tf.float32)

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=.99, epsilon=1e-1)

    best_loss, best_image = float('inf'), None
    configs = {
        'model': model,
        'generated_image': generated_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features,
        'content_weight': content_weight,
        'style_weight': style_weight,
    }

    # norm_means = np.array([103.939, 116.779, 123.68])
    # min_vals = -norm_means
    # max_vals = 255 - norm_means

    tick = datetime.now()
    global_start = datetime.now()

    eval_steps = 20

    for i in range(num_iters):
        gradients, all_loss = compute_grads(configs)
        opt.apply_gradients([(gradients, generated_image)])

        if all_loss[0] < best_loss:
            best_loss = all_loss[0]
            best_image = deprocess(generated_image.numpy())

        if (i + 1) % eval_steps == 0:
            tock = datetime.now()
            plot_image = deprocess(generated_image.numpy())
            display.clear_output(wait=True)
            display_image(plot_image, width=12)

            print("Iteration #{}/{} | Loss: {:.4e} | Style loss: {:.4e} | Content loss: {:.4e} | Time: {:.3f}s".format(
                i + 1, num_iters,
                *all_loss,
                (tock - tick).total_seconds() / eval_steps
            ))
            tick = tock
    print("Iteration #{0}/{0} | Total Time: {1:.4f}s".format(num_iters, (datetime.now() - global_start).total_seconds()))

    return best_image, best_loss


if __name__ == '__main__':
    content_path = 'assets/chengdu.jpg'
    style_path = 'assets/starry_night.jpg'
    content_weight = 1e3
    style_weight = 1e-2

    best_image, best_loss = run(
        content_path, style_path,
        num_iters=200, lr=5,
        content_weight=content_weight, style_weight=style_weight
    )
    print("The best loss is: {:.5e}".format(best_loss))
    display_image(best_image, width=20)
