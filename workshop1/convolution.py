import cv2
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def conv(image, kernel, stride=1, padding='valid'):
    from skimage.exposure import rescale_intensity

    n, m = image.shape[:2]
    k = kernel.shape[0]
    pad = (k - 1) // 2
    w, h = (n - k + 1) // stride, (m - k + 1) // stride
    if padding == 'same':
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, 0)
        w, h = w + 2 * pad // stride, h + 2 * pad // stride
    output = np.zeros((w, h), dtype='float32')
    for y in np.arange(pad, image.shape[0] - pad, stride):
        for x in np.arange(pad, image.shape[1] - pad, stride):
            batch = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
            convolve = (kernel * batch).sum()
            output[(y - pad) // stride, (x - pad) // stride] = convolve
    output = rescale_intensity(output, in_range=(0, 255))
    return (output * 255).astype('uint8')


if __name__ == '__main__':
    kernel_size = 3
    # 模糊
    blur_kernel = np.ones(shape=(kernel_size, kernel_size)) / kernel_size ** 2

    # 竖直边缘检测
    vedge_kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # 水平边缘检测
    hedge_kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    # 浮雕特征检测
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])

    image = cv2.imread('assets/panda.jpg')[:, :, 0]

    blurry = conv(image, blur_kernel, stride=2, padding='same')
    print("original image shape: {}, convolved image shape: {}".format(image.shape, blurry.shape))

    plt.figure()
    plt.title('Blur')
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(blurry, cmap='gray')
    plt.show()

    vedge = conv(blurry, vedge_kernel)
    hedge = conv(blurry, hedge_kernel)
    edge = (vedge + hedge) / 2
    plt.figure()
    plt.title('Horizontal and vertical edge detection')
    plt.subplot(1, 2, 1)
    plt.imshow(blurry, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(edge, cmap='gray')
    plt.show()

    emboss_orig = conv(image, emboss_kernel, stride=1, padding='same')
    emboss_blur = conv(blurry, emboss_kernel, stride=1, padding='same')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Emboss original")
    plt.imshow(emboss_orig, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Emboss blurry")
    plt.imshow(emboss_blur, cmap='gray')
    plt.show()