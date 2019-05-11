import os

import cv2
import matplotlib
import numpy as np
from keras import Input, Model, optimizers
from keras.applications import InceptionV3, inception_v3
from keras.engine.saving import load_model
from keras.layers import Dense, regularizers, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def normalize_image(images):
    return images / 127.5 - 1


def restore_image(images):
    return (images + 1) / 2


def display_image(*images, col=None):
    if col is None:
        col = len(images)
    row = np.math.ceil(len(images) / col)
    plt.figure(figsize=(20, int(20 * col / row)))
    for i, image in enumerate(images):
        image = image.squeeze()
        plt.subplot(row, col, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()


def get_image_loader(image_size, batch_size, color_mode='rgb', shuffle=True):
    image_root = "/Users/ethan/datasets/marvel"
    data_gen = ImageDataGenerator(
        # rescale=1 / 255.0,
        preprocessing_function=inception_v3.preprocess_input
    )
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
    btnk = base.get_layer('mixed8').output
    features = GlobalAveragePooling2D()(btnk)
    return Model(base.input, features)


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
    inputs = Input(shape=(input_size,))
    x = Dropout(.6)(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.6)(x)
    out = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(.1))(x)
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

    print("\n[info] 创建末端全连通神经网络模型并加载优化算法...")
    model = create_nn(x_train.shape[1], y_train.shape[1])
    opt = optimizers.Adam(lr=.0001, beta_1=.95, beta_2=.999, epsilon=1e-8)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    print("\n[info] 开始训练...")
    H = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_valid, y_valid), verbose=1)

    print("\n[info] 训练完成! 开始检测模型性能...")
    test_results = model.predict(x_valid)
    id2label = {val: key for key, val in valid.class_indices.items()}

    print(classification_report(
        y_valid.argmax(axis=1),
        test_results.argmax(axis=1),
        target_names=id2label.values()
    ))

    print("\n[info] 创建完整模型...")
    bottleneck = feature_extractor.output
    out = model(bottleneck)
    hero_recognizer = Model(feature_extractor.input, out)
    print("\n[info] 创建成功! 保存模型到本地...")
    model_file_name = "hero_recognizer.h5"
    hero_recognizer.save(model_file_name)
    print("\n[info] 保存完成, 文件名: {}".format(model_file_name))

    print('\n[info] 从文件中加载模型...')
    new_model = load_model('hero_recognizer.h5')
    print("\n[info] 加载完成, 读取测试图片并开始预测...")
    test_image = cv2.imread('assets/test0.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(test_image, (224, 224)) / 127.5 - 1

    pred = new_model.predict(np.array([input_image]))

    predicted_hero = id2label[pred.argmax()]
    print("\n[info] 模型认为这张图片为[{}].".format(predicted_hero))
    display_image(test_image)
    print('\n[info] 脚本执行完毕!')
