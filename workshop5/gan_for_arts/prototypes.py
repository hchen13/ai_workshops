import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers import Concatenate, Dense, Reshape, LeakyReLU, \
    BatchNormalization, Flatten, Conv2D, Conv2DTranspose


class GatedNonlinearity(Layer):
    def __init__(self, **kwargs):
        super(GatedNonlinearity, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        base, condition = inputs
        a, b = base[:, :, :, ::2], base[:, :, :, 1::2]
        c, d = condition[:, :, :, ::2], condition[:, :, :, 1::2]
        a = a + c
        b = b + d
        result = tf.sigmoid(a) * tf.tanh(b)
        return result


class BaseModel:
    def __init__(self):
        self._path = None
        self.model = None

    def save(self, filename):
        fullname = os.path.join(self._path, filename)
        if not fullname.endswith('.h5'):
            fullname += '.h5'
        self.model.save_weights(fullname)

    def load(self, filename):
        fullname = os.path.join(self._path, filename)
        if not fullname.endswith('.h5'):
            fullname += '.h5'
        try:
            self.model.load_weights(fullname)
        except (NotFoundError, OSError):
            print("[ERROR] loading weights failed: {}".format(filename))
            return False
        return True


class Generator(BaseModel):
    def __init__(self, num_classes, target_width, bn=False, noise_size=128, min_channels=64, max_channels=1024):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.target_width = target_width
        self.noise_size = noise_size
        self.bn = bn
        self.create_model(min_channels, max_channels)
        self._path = './.models/generator/'
        os.makedirs(self._path, exist_ok=True)

    def create_model(self, min_channels, max_channels):
        gate = GatedNonlinearity()
        _width = 4
        _num_blocks = np.log2(self.target_width / _width).astype(np.uint8)
        _num_channels = min(max_channels, min_channels * (2 ** _num_blocks))
        _num_channels = max(min_channels, _num_channels)
        noise = Input(shape=[self.noise_size], dtype=tf.float32)
        label = Input(shape=[self.num_classes], dtype=tf.float32)
        x = Concatenate()([noise, label])
        output = Dense(_width * _width * _num_channels)(x)
        output = Reshape(target_shape=(_width, _width, _num_channels))(output)

        for i in range(1, _num_blocks + 1):
            if self.bn:
                output = BatchNormalization()(output)
            condition = Dense(_width * _width * _num_channels, use_bias=False)(label)
            condition = Reshape(target_shape=(_width, _width, _num_channels))(condition)
            output = gate([output, condition])
            _num_channels = min(max_channels, min_channels * (2 ** (_num_blocks - i)))
            _num_channels = max(min_channels, _num_channels)
            if i == _num_blocks:
                _num_channels = 3
            _width *= 2
            output = Conv2DTranspose(_num_channels, kernel_size=5, strides=2, padding='same')(output)
        gen_image = tf.tanh(output)
        self.model = Model(inputs=[noise, label], outputs=[gen_image])


class Discriminator(BaseModel):
    def __init__(self, num_classes, image_width, depth, leak=.2, bn_epsilon=0):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.image_width = image_width
        self.leak = leak
        self.bn_epsilon = bn_epsilon
        self.create_model(depth)

        self._path = './.models/discriminator/'
        os.makedirs(self._path, exist_ok=True)

    def create_model(self, depth):
        def _conv(d):
            return Conv2D(d, kernel_size=5, strides=2, padding='same')

        _num_blocks = np.log2(self.image_width / 4).astype(np.uint8) - 1
        image = Input(shape=(self.image_width, self.image_width, 3), dtype=tf.float32)
        conv = _conv(depth)(image)
        conv = LeakyReLU(alpha=self.leak)(conv)
        for i in range(_num_blocks):
            depth *= 2
            conv = _conv(depth)(conv)
            if self.bn_epsilon != 0:
                conv = BatchNormalization(epsilon=self.bn_epsilon)(conv)
            conv = LeakyReLU(alpha=self.leak)(conv)
        features = Flatten()(conv)
        logit = Dense(1, name='wasserstein_logit')(features)
        label = Dense(self.num_classes, activation='softmax', name='label')(features)
        self.model = Model(inputs=[image], outputs=[logit, label])

    @tf.function
    def _pretrain_step(self, x, y_true, opt):
        with tf.GradientTape() as tape:
            _, y = self.model(x)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y))
            acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_true, y))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, acc

    def pretrain(self, train_data, valid_data, iterations, learning_rate, retrain=False, verbose=True):

        def _eval():
            y_true, y_pred = [], []
            for i in range(len(valid_data)):
                _x, _y = valid_data[i]
                _, y = self.model.predict(_x)
                y_true.append(_y)
                y_pred.append(y)
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            report = classification_report(
                y_true.argmax(axis=1),
                y_pred.argmax(axis=1),
                target_names=valid_data.class_indices.keys()
            )
            print('\n', report)

        model_name = 'pretrain{}'.format(iterations)
        if self.bn_epsilon == 0:
            model_name = 'no_bn_' + model_name

        if not retrain:
            success = self.load(model_name)
            if success:
                return
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for i in range(iterations):
            images, labels = next(train_data)
            loss, acc = self._pretrain_step(images, labels, optimizer)
            if verbose and ( (i + 1) % 3 == 0 or i == iterations - 1 ):
                print("\rIteration #{}: train_loss = {:.3f} | train_acc = {:.1f}%".format(i + 1, loss, acc * 100))
        _eval()
        self.save(model_name)


if __name__ == '__main__':
    num_classes = 14
    minimum_feature_channels = 64
    target_image_width = 128
    disc = Discriminator(num_classes=num_classes, image_width=target_image_width, depth=minimum_feature_channels)
