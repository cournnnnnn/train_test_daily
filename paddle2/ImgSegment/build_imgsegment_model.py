"""
图像分割是对图像中的每个像素加标签的一个过程
使得具有相同标签的像素具有某种共同视觉特性
应用领域，无人车、地块检测、表计识别等等
"""
import random
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import transforms
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from paddle.nn import functional as F

class ImgTranspose(object):
    """
    图像预处理工具，用于将Mask图像进行升维(160, 160) => (160, 160, 1)，
    并对图像的维度进行转换从HWC变为CHW
    """
    def __init__(self, fmt):
        self.format = fmt

    def __call__(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        return img.transpose(self.format)

class PetDataset(Dataset):
    """
    数据集定义
    """
    def __init__(self, image_path, label_path, mode='train'):
        """
        构造函数
        """
        self.image_size = (160, 160)
        self.image_path = image_path
        self.label_path = label_path
        self.mode = mode.lower()
        self.eval_image_num = 1000

        assert self.mode in ['train', 'test'], \
            "mode should be 'train' or 'test', but got {}".format(self.mode)

        self._parse_dataset()

        self.transforms = transforms.Compose([
            ImgTranspose((2, 0, 1))
        ])

    def _sort_images(self, image_dir, image_type):
        """
        对文件夹内的图像进行按照文件名排序
        """
        files = []

        for image_name in os.listdir(image_dir):
            if image_name.endswith('.{}'.format(image_type)) \
                    and not image_name.startswith('.'):
                files.append(os.path.join(image_dir, image_name))

        return sorted(files)

    def _parse_dataset(self):
        """
        由于所有文件都是散落在文件夹中，在训练时我们需要使用的是数据集和标签对应的数据关系，
        所以我们第一步是对原始的数据集进行整理，得到数据集和标签两个数组，分别一一对应。
        这样可以在使用的时候能够很方便的找到原始数据和标签的对应关系，否则对于原有的文件夹图片数据无法直接应用。
        在这里是用了一个非常简单的方法，按照文件名称进行排序。
        因为刚好数据和标签的文件名是按照这个逻辑制作的，名字都一样，只有扩展名不一样。
        """
        temp_train_images = self._sort_images(self.image_path, 'jpg')
        temp_label_images = self._sort_images(self.label_path, 'png')

        random.Random(1337).shuffle(temp_train_images)
        random.Random(1337).shuffle(temp_label_images)

        if self.mode == 'train':
            self.train_images = temp_train_images[:-self.eval_image_num]
            self.label_images = temp_label_images[:-self.eval_image_num]
        else:
            self.train_images = temp_train_images[-self.eval_image_num:]
            self.label_images = temp_label_images[-self.eval_image_num:]

    def _load_img(self, path, color_mode='rgb'):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        with open(path, 'rb') as f:
            img = Image.open(paddle.io.BytesIO(f.read()))
            if color_mode == 'grayscale':
                # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
                # convert it to an 8-bit grayscale image.
                if img.mode not in ('L', 'I;16', 'I'):
                    img = img.convert('L')
            elif color_mode == 'rgba':
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
            elif color_mode == 'rgb':
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')

            if self.image_size is not None:
                if img.size != self.image_size:
                    img = img.resize(self.image_size, Image.NEAREST)

            return img

    def __getitem__(self, idx):
        """
        返回 image, label
        """
        # 花了比较多的时间在数据处理这里，需要处理成模型能适配的格式，踩了一些坑（比如有不是RGB格式的）
        # 有图片会出现通道数和期望不符的情况，需要进行相关考虑

        # 加载原始图像
        train_image = self._load_img(self.train_images[idx])
        x = np.array(train_image, dtype='float32')

        # 对图像进行预处理，统一大小，转换维度格式（HWC => CHW）
        x = self.transforms(x)

        # 加载Label图像
        label_image = self._load_img(self.label_images[idx], color_mode="grayscale")
        y = np.array(label_image, dtype='uint8')

        # 图像预处理
        # Label图像是二维的数组(size, size)，升维到(size, size, 1)后才能用于最后loss计算
        y = self.transforms(y)

        # 返回img, label，转换为需要的格式
        return x, y.astype('int64')

    def __len__(self):
        """
        返回数据集总数
        """
        return len(self.train_images)

"""
继承paddle.nn.Layer自定义了一个SeparableConv2d Layer类，
整个过程是把filter_size * filter_size * num_filters的Conv2d操作拆解为两个子Conv2d，
先对输入数据的每个通道使用filter_size * filter_size * 1的卷积核进行计算，输入输出通道数目相同，
之后在使用1 * 1 * num_filters的卷积核计算
"""
class SeparableConv2d(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(SeparableConv2d, self).__init__()
        # 第一次卷积操作没有偏置参数
        self.conv_1 = paddle.nn.Conv2d(in_channels,
                                       in_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=in_channels,
                                       weight_attr=weight_attr,
                                       bias_attr=False,
                                       data_format=data_format)
        self.pointwise = paddle.nn.Conv2d(in_channels,
                                          out_channels,
                                          1,
                                          stride=1,
                                          padding=0,
                                          dilation=1,
                                          groups=1,
                                          weight_attr=weight_attr,
                                          data_format=data_format)

    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.pointwise(y)

        return y

"""
下采样是有一个模型逐渐向下画曲线的一个过程，
这个过程中是不断的重复一个单元结构将通道数不断增加，形状不断缩小，并且引入残差网络结构，
"""
class Encoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.relu = paddle.nn.ReLU()
        self.separable_conv_01 = SeparableConv2d(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 padding='same')
        self.bn = paddle.nn.BatchNorm2d(out_channels)
        self.separable_conv_02 = SeparableConv2d(out_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 padding='same')
        self.pool = paddle.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_conv = paddle.nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=2,
                                              padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relu(inputs)
        y = self.separable_conv_01(y)
        y = self.bn(y)
        y = self.relu(y)
        y = self.separable_conv_02(y)
        y = self.bn(y)
        y = self.pool(y)

        residual = self.residual_conv(previous_block_activation)
        y = paddle.add(y, residual)

        return y

"""
进行上采样，通道数逐渐减小，
对应图片尺寸逐步增加，直至恢复到原图像大小
"""
class Decoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.relu = paddle.nn.ReLU()
        #转置卷积的计算过程相当于卷积的反向计算
        self.conv_transpose_01 = paddle.nn.ConvTranspose2d(in_channels,
                                                           out_channels,
                                                           kernel_size=3,
                                                           padding='same')
        self.conv_transpose_02 = paddle.nn.ConvTranspose2d(out_channels,
                                                           out_channels,
                                                           kernel_size=3,
                                                           padding='same')
        self.bn = paddle.nn.BatchNorm2d(out_channels)

        self.upsample = paddle.nn.Upsample(scale_factor=2.0)
        self.residual_conv = paddle.nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relu(inputs)
        y = self.conv_transpose_01(y)
        y = self.bn(y)
        y = self.relu(y)
        y = self.conv_transpose_02(y)
        y = self.bn(y)
        y = self.upsample(y)

        residual = self.upsample(previous_block_activation)
        residual = self.residual_conv(residual)

        y = paddle.add(y, residual)

        return y

"""
照U型网络结构格式进行整体的网络结构搭建，
三次下采样，四次上采样。
"""
class PetNet(paddle.nn.Layer):
    def __init__(self, num_classes):
        super(PetNet, self).__init__()

        self.conv_1 = paddle.nn.Conv2d(3, 32,
                                       kernel_size=3,
                                       stride=2,
                                       padding='same')
        #根据当前批次数据按通道计算的均值和方差进行归一化
        self.bn = paddle.nn.BatchNorm2d(32)
        self.relu = paddle.nn.ReLU()

        in_channels = 32
        self.encoders = []
        self.encoder_list = [64, 128, 256]
        self.decoder_list = [256, 128, 64, 32]

        # 根据下采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.encoder_list:
            block = self.add_sublayer('encoder_%s'.format(out_channels),
                                      Encoder(in_channels, out_channels))
            self.encoders.append(block)
            in_channels = out_channels

        self.decoders = []

        # 根据上采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.decoder_list:
            block = self.add_sublayer('decoder_%s'.format(out_channels),
                                      Decoder(in_channels, out_channels))
            self.decoders.append(block)
            in_channels = out_channels

        self.output_conv = paddle.nn.Conv2d(in_channels,
                                            num_classes,
                                            kernel_size=3,
                                            padding='same')

    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.bn(y)
        y = self.relu(y)

        for encoder in self.encoders:
            y = encoder(y)

        for decoder in self.decoders:
            y = decoder(y)

        y = self.output_conv(y)

        return y


class SoftmaxWithCrossEntropy(paddle.nn.Layer):
    def __init__(self):
        super(SoftmaxWithCrossEntropy, self).__init__()

    def forward(self, input, label):
        loss = F.softmax_with_cross_entropy(input,
                                            label,
                                            return_softmax=False,
                                            axis=1)
        return paddle.mean(loss)


def showimg(image,label):
    # 进行图片的展示
    plt.figure()

    plt.subplot(1,2,1),
    plt.title('Train Image')
    plt.imshow(image.transpose((1, 2, 0)).astype('uint8'))
    plt.axis('off')

    plt.subplot(1,2,2),
    plt.title('Label')
    plt.imshow(np.squeeze(label, axis=0).astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':

    ##下载数据太慢！！！！
    train_images_path =''
    label_images_path =''
    num_classes = 37
    # 训练数据集
    train_dataset = PetDataset(train_images_path, label_images_path, mode='train')

    # 验证数据集
    val_dataset = PetDataset(train_images_path, label_images_path, mode='test')

    # # 抽样一个数据
    # image, label = train_dataset[0]
    # showimg(image,label)
    model = paddle.Model(PetNet(num_classes))
    optim = paddle.optimizer.RMSProp(learning_rate=0.001,
                                     rho=0.9,
                                     momentum=0.0,
                                     epsilon=1e-07,
                                     centered=False,
                                     parameters=model.parameters())

    model.prepare(optimizer=optim, loss=SoftmaxWithCrossEntropy())
    model.fit(train_dataset,
              val_dataset,
              epochs=15,
              batch_size=32,save_dir='./model/final',save_freq=1)
