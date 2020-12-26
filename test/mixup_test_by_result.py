"""
从结果上看，mixup是
两张图片的叠加，factor是图片的虚化度
作用：数据增强，弥补训练图像数据集不足，达到对训练数据扩充的目的
同类增强（如：翻转、旋转、缩放、移位、模糊等）和混类增强（如mixup）两种方式
"""



import cv2
from matplotlib import pyplot as plt
import numpy as np
img_file1 = '/home/chan/IdeaProjects/train_test_daily/jpg/a_0.jpg'
img_file2 = '/home/chan/IdeaProjects/train_test_daily/jpg/a_1.jpg'

img1 = cv2.imread(img_file1)
img2 = cv2.imread(img_file2)

alpha = 0.5
beta = 0.5
factor = np.random.beta(alpha, beta)
factor = max(0.0, min(1.0, factor))


def _mixup_img(img1, img2, factor):
    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])
    img = np.zeros((h, w, img1.shape[2]), 'float32')
    img[:img1.shape[0], :img1.shape[1], :] = \
        img1.astype('float32') * factor
    img[:img2.shape[0], :img2.shape[1], :] += \
        img2.astype('float32') * (1.0 - factor)
    return img.astype('uint8')

if __name__ == '__main__':
    img = _mixup_img(img1, img2, factor)
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.title('1 Image')
    plt.imshow(np.array(img1).astype('uint8'))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('2 Image')
    plt.imshow(np.array(img2).astype('uint8'))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('mix_up Image')
    plt.imshow(np.array(img).astype('uint8'))
    plt.axis('off')

    plt.show()

    plt.figure()
    plt.title('mix_up Image')
    plt.imshow(np.array(img).astype('uint8'))
    plt.axis('off')

    plt.show()