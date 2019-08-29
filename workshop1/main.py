import glob
import os

import cv2
import matplotlib
import numpy as np
from keras import Input, Model, optimizers
from keras.applications import InceptionV3, inception_v3
from keras.engine.saving import load_model
from keras.layers import Dense, regularizers, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
from sklearn.metrics import classification_report

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def normalize_image(images):
    """
    将图像像素值从[0,255]范围缩放到[-1,1]
    :param images: numpy张量, 一般为4维张量 image[num_images][image_height][image_width][3]
    :return: 缩放后的张量, shape不变
    """
    return images / 127.5 - 1


def restore_image(images):
    """
    将图像像素值从[-1,1]缩放到[0,1]
    :param images: numpy张量, 一般为4维张量 image[num_images][image_height][image_width][3]
    :return: 缩放后的张量, shape不变
    """
    return (images + 1) / 2


def display_image(*images, col=None):
    """
    可视化任意数量的图片, 调用方式:
    display_image(image1, image2, image3, image4, col=2)
    假设我们有image1-4四个变量, 每个变量都是图片张量, 以上调用会将4张图片以2x2的排列方式绘制出来

    :param images: 可传入任意数量的参数, 每个参数是一张图片的numpy三维张量, 即image[height][width][3]
    :param col: 显示多张图片时一行最多显示多少张
    :return: None
    """
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
    """
    从目录创建两个图像读取的生成器(generator), 分别用于分批次读取训练集和测试集的图像数据
    从生成器中读取到的图片成为统一尺寸
    :param image_size: 图片的统一边长
    :param batch_size: 每一批次中含有图片个数
    :param color_mode: 颜色模式, 默认为rgb彩色, 可以选择的还有'grayscale'
    :param shuffle: 是否打乱图片顺序
    :return: train set和validation set的图片生成器
    """
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


def get_feature_extractor(image_size=224):
    """
    以Google的预训练Inception v3模型为基础创建图像特征提取器
    用法:
    model = get_feature_extractor(image_size=299)
    features = model.predict([image1, image2])
    其中得到的features为2x1280的矩阵, 矩阵中的每一行分别为image1和image2的特征

    :return: keras Model对象, 可直接使用该模型对象对图片进行特征提取
    """
    base = InceptionV3(include_top=False, input_shape=(image_size, image_size, 3), pooling='avg')
    btnk = base.get_layer('mixed8').output
    features = GlobalAveragePooling2D()(btnk)
    return Model(base.input, features)


def extract_features(extractor, dataset):
    """
    用extractor来提取dataset中的特征
    :param extractor: get_feature_extractor()函数中创建的提取器
    :param dataset: 某个图像读取生成器
    :return: x和y, x为特征矩阵, y为图像标签矩阵, dataset中的图片个数为行数, x的列数为extractor提取的特征个数, y的列数为标签个数
    """
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
    """
    创建新的全连通神经网络模型
    :param input_size: 全连通神经网络的输入数据个数, 在这里保持与feature_extractor提取的特征个数一致
    :param num_classes: 全连通神经网络的输出节点数, 为待分类的标签个数
    :return: 全新的神经网络模型
    """
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