import scipy

from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from keras.applications import inception_v3, InceptionV3
from keras.preprocessing.image import load_img, img_to_array

settings = {
    'features': {
        'mixed2': .2,
        'mixed3': .5,
        'mixed4': 2.,
        'mixed5': 1.5,
    }
}


bp = '/Users/ethan/Pictures/StyleTransfer/chengdu.jpg'

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

def preprocess_image(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = inception_v3.preprocess_input(image)
    return image


def deprocess_image(image):
    x = image.copy()
    x = x.squeeze()
    x /= 2.
    x += .5
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


K.set_learning_phase(0)

model = InceptionV3(include_top=False)
dream = model.input
print("Model loaded.")

layer_dict = {layer.name: layer for layer in model.layers}

loss = K.variable(0.)
for layer_name in settings['features']:
    if layer_name not in layer_dict:
        raise ValueError('Layer {} not found.'.format(layer_name))

    layer_weight = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(x), 'float32'))

    loss += layer_weight * K.sum(K.square(x)) / scaling

grads = K.gradients(loss, dream)[0]
grads /= K.mean(K.abs(grads)) + K.epsilon()
fetch_loss_grads = K.function([dream], [loss, grads])


def eval_loss_and_grads(x):
    outs = fetch_loss_grads([x])
    loss_val = outs[0]
    grads_val = outs[1]
    return loss_val, grads_val


def resize_image(image, size):
    image = np.copy(image)
    factor = (
        1,
        float(size[0]) / image.shape[1],
        float(size[1]) / image.shape[2],
        1
    )
    return scipy.ndimage.zoom(image, factor, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_val, grad_val = eval_loss_and_grads(x)
        if max_loss is not None and loss_val > max_loss:
            break
        x += step * grad_val
    return x


step = .01
num_octave = 3
octave_scale = 1.4
iterations = 20
max_loss = 10.


if __name__ == '__main__':
    image = preprocess_image(bp)
    original_shape = image.shape[1:3]

    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_image = np.copy(image)
    shrunk_original_image = resize_image(image, successive_shapes[0])

    for shape in successive_shapes:
        print("Processing image shape", shape)
        image = resize_image(image, shape)
        continue
        image = gradient_ascent(image, iterations=iterations, step=step, max_loss=max_loss)
        upscaled = resize_image(shrunk_original_image, shape)
        downscaled = resize_image(original_image, shape)
        lost_detail = downscaled - upscaled
        image += lost_detail
        shrunk_original_image = resize_image(original_image, shape)


