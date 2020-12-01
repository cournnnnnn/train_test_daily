import paddle
import os
import cv2
import numpy as np
import random
import pandas as pd
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.io import Dataset
from datetime import datetime
from paddle.static import InputSpec

train_datapath = '/home/chan/dataset/eye/PALM-Training400/PALM-Training400'
valid_datapath = '/home/chan/dataset/eye/PALM-Validation400'
valid_filepath = '/home/chan/dataset/eye/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx'

# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img

class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()

        self.data = []
        global train_datapath
        global valid_datapath
        global valid_filepath

        if mode == 'train':
            files = os.listdir(train_datapath)
            random.shuffle(files)       #洗一下
            for name in files:
                filepath = os.path.join(train_datapath, name)
                img = cv2.imread(filepath)
                img = transform_img(img)
                if name[0] == 'H' or name[0] == 'N':
                    label = 0
                elif name[0] == 'P':
                    label = 1
                else:
                    raise('Not excepted file name')
                self.data.append([img,label])
            # self.data = [
            #     ['traindata1', 'label1'],
            #     ['traindata2', 'label2'],
            #     ['traindata3', 'label3'],
            #     ['traindata4', 'label4'],
            # ]
        else:
            data = pd.read_excel(valid_filepath,sheet_name=0,header=0,index_col=0)
            for i in range(data.shape[0]):
                imgfile = os.path.join(valid_datapath,data['imgName'].iloc[i])
                img = cv2.imread(imgfile)
                img = transform_img(img)
                label = data['Label'].iloc[i]
                self.data.append([img,label])

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = self.data[index][0]
        label = self.data[index][1]

        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)

# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self,num_class=1):
        super(LeNet, self).__init__()
        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.conv1 = paddle.nn.Conv2D(in_channels=3,out_channels=6,kernel_size=5)
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2,stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6,out_channels=16,kernel_size=5)
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2,stride=2)
        # 创建第3个卷积层
        self.conv3 = paddle.nn.Conv2D(in_channels=16,out_channels=120,kernel_size=4)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc1 = paddle.nn.Linear(in_features=300000,out_features=64)
        self.fc2 = paddle.nn.Linear(in_features=64,out_features=num_class)
    # 网络的前向计算过程
    def forward(self,x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = self.pool2(x)
        x = F.sigmoid(x)
        x = self.conv3(x)
        x = paddle.flatten(x,start_axis=1,stop_axis=-1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x


def main_fit():
    # 测试定义的数据集
    sr_time = datetime.now()
    train_dataset = MyDataset(mode='train')
    train_dataset = paddle.io.DataLoader(train_dataset,batch_size=20)
    valid_dataset = MyDataset(mode='valid')
    er_time = datetime.now()
    print('total cost time of read data: {} seconds'.format((er_time-sr_time).seconds))

    input = InputSpec([None,3,32,32],dtype='float32',name='image')
    label = InputSpec([None,1],dtype='int64',name='label')

    #定义网络
    model = paddle.Model(LeNet(num_class=2),input,label)
    opt = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters())
    model.prepare(opt,loss=paddle.nn.CrossEntropyLoss(),metrics=Accuracy())

    start_time = datetime.now()
    #训练模型
    model.fit(train_dataset,batch_size=10,epochs=2,verbose=1,save_dir='./output1',save_freq=1)
    print('===================================================================================================')
    model.evaluate(valid_dataset,batch_size=10,verbose=1)
    end_time = datetime.now()
    print('fit and eval cost time : {} seconds'.format((end_time-start_time).seconds))


if __name__ == '__main__':

    main_fit()


