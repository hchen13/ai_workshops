import os

import cv2
import matplotlib
import numpy as np
from keras import Input, Model, optimizers
from keras.applications import InceptionV3
from keras.engine.saving import load_model
from keras.layers import Dense, regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

def display_image(*images, col=None):
    if col is None:
        col = len(images)
    plt.figure(figsize=(16, 9))
    row = np.math.ceil(len(images) / col)
    for i, image in enumerate(images):
        image = image.squeeze()
        plt.subplot(row, col, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()


def get_image_loader(image_size, batch_size, color_mode='rgb', shuffle=True):
    image_root = "/Users/ethan/datasets/marvel"
    data_gen = ImageDataGenerator(rescale=1 / 255.0)
    train_loader = data_gen.flow_from_directory(
        os.path.join(image_root, 'train'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    valid_loader = data_gen.flow_from_directory(
        os.path.join(image_root, 'valid'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return train_loader, valid_loader


def get_feature_extractor():
    base = InceptionV3(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return base


def extract_features(extractor, dataset):
    x, y = None, None
    n = len(dataset)
    for i in range(n):
        print("\r[info] 正在提取特征: batch #{}/{}...".format(i + 1, n), end='')
        batch_x, batch_y = dataset[i]
        features = extractor.predict(batch_x)
        if x is None:
            x = features
            y = batch_y
        else:
            x = np.vstack([x, features])
            y = np.vstack([y, batch_y])
    return x, y


def create_nn(input_size, num_classes):
    inputs = Input(shape=(input_size, ))
    out = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(.01))(inputs)
    return Model(inputs, out)


if __name__ == '__main__':
    train, valid = get_image_loader(image_size=224, batch_size=16, shuffle=False)
    feature_extractor = get_feature_extractor()
    print("\n[info] 训练集特征提取:")
    x_train, y_train = extract_features(feature_extractor, train)
    print("\n[info] 训练集完成!\n")

    print("\n[info] 测试集特征提取:")
    x_valid, y_valid = extract_features(feature_extractor, valid)
    print("\n[info] 测试集完成!\n")
    model = create_nn(2048, 7)
    opt = optimizers.adam(lr=.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    H = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_valid, y_valid), verbose=1)

    test_results = model.predict(x_valid)

    id2label = {val: key for key, val in valid.class_indices.items()}

    print(classification_report(
        y_valid.argmax(axis=1),
        test_results.argmax(axis=1),
        target_names=id2label.values()
    ))

    bottleneck = feature_extractor.output
    out = model(bottleneck)
    hero_recognizer = Model(feature_extractor.input, out)
    hero_recognizer.save("hero_recognizer.h5")

    new_model = load_model('hero_recognizer.h5')
    test_image = cv2.imread('assets/test2.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(test_image, (224, 224)) / 255.0

    pred = new_model.predict(np.array([input_image]))

    predicted_hero = id2label[pred.argmax()]
    print("The model thinks this image is of [{}].".format(predicted_hero))
    display_image(test_image)
