import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from keras import Sequential, optimizers
from keras.layers import Dense, Dropout, regularizers


def load_raw_data(file_path="~/datasets/housing/housing.csv"):
    """
    从csv文件加载原始数据集
    """
    headers = [
        "销售日期", "销售价格", "卧室数",
        "浴室数", "房屋面积", '停车面积',
        "楼层数", "房屋评分", "建筑面积",
        "地下室面积", "建筑年份", "修复年份",
        '纬度', '经度'
    ]
    data = pd.read_csv(file_path, header=None, names=headers)
    return data


def train_test_split(data, test_ratio=0.1):
    """
    将数据以一定比例分割成训练集与测试集
    """
    test_size = int(len(data) * test_ratio)
    train_size = len(data) - test_size

    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


def input_output_split(dataset):
    """
    将销售价格从原始数据中提出来作为输出, 剩余部分作为输入
    """
    y = dataset[['销售价格']]
    x = dataset.iloc[:, dataset.columns != '销售价格']
    return x, y


def preprocess_data(x, y):
    """
    对原始数据进行预处理, 输入数据进行归一化(normalization), 输出数据除以一万
    """
    def norm(x):
        return (x - x.mean()) / x.std()

    x = norm(x)
    y = y / 10000
    return x.values, y.values


def create_shallow_nn():
    """
    创建只有一层隐藏层的浅层神经网络
    """
    model = Sequential()
    model.add(Dense(20, input_shape=(13,), activation='relu'))
    model.add(Dense(1, activation='relu'))
    opt = optimizers.adam(lr=.001)
    model.compile(optimizer=opt, loss='mse')
    return model


def create_deep_nn():
    """
    创建拥有3个隐藏层和一些正则化(regularization)方法的深度神经网络
    """
    model = Sequential()
    model.add(Dense(50, input_shape=(13,), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(1, activation='relu', kernel_regularizer=regularizers.l2()))

    opt = optimizers.adam(lr=.001)
    model.compile(optimizer=opt, loss='mse')
    return model


def evaluate_model(model, test_x, test_y):
    preds = model.predict(test_x)
    predicted_prices = preds * 10000
    true_prices = test_y * 10000
    diff = abs(predicted_prices - true_prices)
    count = (diff > 100_000).sum()
    print("{}套房屋预测房价与真实房价总差额: ${:,.2f}".format(len(diff), diff.sum()))
    print("平均误差为${:,.2f}".format(diff.sum() / len(diff)))
    print("其中有{}套房屋预测价格与真实房价差距大于10万美金".format(count))

    # 取前100套房屋, 将其绘制成图
    sample_preds = predicted_prices[:100]
    sample_truth = true_prices[:100]
    plt.figure(figsize=(24, 9))
    plt.plot(sample_truth, label='true prices')
    plt.plot(sample_preds, label='predicted prices')
    plt.ylim(bottom=0)
    plt.legend()
    for i, d in enumerate(diff[:100]):
        d = d[0]
        if d <= 100_000:
            continue
        plt.text(i, max(sample_preds[i], sample_truth[i]), "{:,.2f}".format(d))
    plt.show()


if __name__ == '__main__':
    data = load_raw_data("data/housing.csv")
    train, test = train_test_split(data, test_ratio=.1)
    train_x, train_y = input_output_split(train)
    test_x, test_y = input_output_split(test)
    train_x, train_y = preprocess_data(train_x, train_y)
    test_x, test_y = preprocess_data(test_x, test_y)

    shallow = create_shallow_nn()
    shallow.fit(train_x, train_y, epochs=50, batch_size=16, verbose=2)
    print("训练完成!\n")
    evaluate_model(shallow, test_x, test_y)

    deep = create_deep_nn()
    deep.fit(train_x, train_y, epochs=50, batch_size=16, verbose=2)
    print("训练完成!\n")
    evaluate_model(deep, test_x, test_y)