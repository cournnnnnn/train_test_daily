"""
图片搜索
其基本思路是，先将图片使用卷积神经网络转换为高维空间的向量表示，然后计算两张图片的高维空间的向量表示之间的相似程度(本示例中，我们使用余弦相似度)。
在模型训练阶段，其训练目标是让同一类别的图片的相似程度尽可能的高，不同类别的图片的相似程度尽可能的低。
在模型预测阶段，对于用户上传的一张图片，会计算其与图片库中图片的相似程度，
返回给用户按照相似程度由高到低的图片的列表作为检索的结果。
"""

import paddle
import numpy as np
from PIL import Image
from collections import defaultdict
import random
from paddle.nn import functional as F
from paddle.static import InputSpec


#获取图像数据，图像类别
def trans_dataset(mode):
    dataset = paddle.vision.datasets.cifar.Cifar10(mode=mode)
    data_len = len(dataset)
    x_train = np.zeros((data_len, 3, 32, 32))
    y_train = np.zeros((data_len, 1), dtype='int32')

    for i in range(data_len):
        img,label = dataset[i]
        x_train[i,:,:,:] = img.reshape((3,32,32))/255.
        y_train[i] = label
        y_train = np.squeeze(y_train)

    return x_train,y_train

#绘制图像展示
def show_collage(examples):
    height_width = 32
    box_size = height_width + 2
    num_rows, num_cols = examples.shape[:2]

    collage = Image.new(
        mode="RGB",
        size=(num_cols * box_size, num_rows * box_size),
        color=(255, 255, 255),
    )
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
            array = array.transpose(1,2,0)
            collage.paste(
                Image.fromarray(array), (col_idx * box_size, row_idx * box_size)
            )

    collage = collage.resize((2 * num_cols * box_size, 2 * num_rows * box_size))
    return collage

#读取num批次的2×10×3×32×32的数据，20张图像，每个类别2张
def reader_creator(num_batchs):
    num_classes = 10
    height_width = 32

    x_train,y_train = trans_dataset('train')
    x_test,y_test = trans_dataset('test')
    #类别下面是索引
    class_idx_to_train_idxs = defaultdict(list)
    for y_train_idx, y in enumerate(y_train):
        class_idx_to_train_idxs[y].append(y_train_idx)

    class_idx_to_test_idxs = defaultdict(list)
    for y_test_idx, y in enumerate(y_test):
        class_idx_to_test_idxs[y].append(y_test_idx)

    def reader():
        iter_step = 0
        while True:
            if iter_step >= num_batchs:
                break
            iter_step += 1
            x = np.empty((2, num_classes, 3, height_width, height_width), dtype=np.float32)
            for class_idx in range(num_classes):#类别0～9
                examples_for_class = class_idx_to_train_idxs[class_idx]#train_idx这一类的所有索引
                #随机选取一个类两个图的索引，避免一样
                anchor_idx = random.choice(examples_for_class)
                positive_idx = random.choice(examples_for_class)
                while positive_idx == anchor_idx:
                    positive_idx = random.choice(examples_for_class)
                x[0, class_idx] = x_train[anchor_idx]
                x[1, class_idx] = x_train[positive_idx]
            yield x  #2×10

    return reader

def train(model):
    num_classes = 10
    print('start training ... ')
    model.train()
    pairs_train_reader = reader_creator(100)  #100*2*10*3*32*32
    inverse_temperature = paddle.to_tensor(np.array([1.0/0.2], dtype='float32'))

    epoch_num = 200

    opt = paddle.optimizer.Adam(learning_rate=0.0001,
                                parameters=model.parameters())

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(pairs_train_reader()):
            anchors_data, positives_data = data[0], data[1]
            #anchors,positives-->10*3*32*32
            anchors = paddle.to_tensor(anchors_data)
            positives = paddle.to_tensor(positives_data)
            #anchor_embeddings,positive_embeddings--->10*8
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            #matmul:[10*8]*[8*10]--->[10*10]
            similarities = paddle.matmul(anchor_embeddings, positive_embeddings, transpose_y=True)
            similarities = paddle.multiply(similarities, inverse_temperature)

            sparse_labels = paddle.arange(0, num_classes, dtype='int64')

            loss = F.cross_entropy(similarities, sparse_labels)

            if batch_id % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()
    # 保存模型&参数
    path = "./search_similar/model"
    paddle.jit.save(model,path,input_spec=[InputSpec(shape=[None, 3,32,32], dtype='float32')])

#网络用富含信息的8个数去表现一个图的特征
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=2)

        self.conv2 = paddle.nn.Conv2D(in_channels=32,
                                      out_channels=64,
                                      kernel_size=(3,3),
                                      stride=2)

        self.conv3 = paddle.nn.Conv2D(in_channels=64,
                                      out_channels=128,
                                      kernel_size=(3,3),
                                      stride=2)

        self.gloabl_pool = paddle.nn.AdaptiveAvgPool2D((1,1))

        self.fc1 = paddle.nn.Linear(in_features=128, out_features=8)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.gloabl_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self.fc1(x)
        x = x / paddle.norm(x, axis=1, keepdim=True)
        return x

if __name__ == '__main__':

    print('__')
    # num_classes = 10
    # # print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    # # sample_idxs = np.random.randint(0, 50000, size=(5, 5))
    # # examples = x_train[sample_idxs]
    # # collage = show_collage(examples)
    # # collage.show()
    # # pairs_train_reader = reader_creator(100)  #100*2*10*3*32*32
    # # show_collage(next(pairs_train_reader())).show()
    #
    model = MyNet()
    train(model)









