import paddle
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.static import InputSpec


# train_images = np.zeros((50000, 32, 32, 3), dtype='float32')
# train_labels = np.zeros((50000, 1), dtype='int32')
# for i, data in enumerate(train_data):
#     train_image, train_label = data
#     train_image = train_image.reshape((3, 32, 32 )).astype('float32') / 255.
#     train_image = train_image.transpose(2, 1, 0)  #(3,32,32)放入(32,32,3) 中需要轴变化一下啊
#     #train_image = train_image.transpose(1, 2, 0)
#     train_images[i, :, :, :] = train_image
#     train_labels[i, 0] = train_label
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# plt.figure(figsize=(10,10))
# sample_idxs = np.random.choice(50000, size=25, replace=False) #随机从50000中选择25个(range(0,50000))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[sample_idxs[i]], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[sample_idxs[i]][0]])
# plt.show()

class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(2,2))
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(2,2))

        self.flatten = paddle.nn.Flatten()

        self.linear1 = paddle.nn.Linear(in_features=64*6*6, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=10)

    def forward(self, x):                #[3,32,32]
        x = self.conv1(x)                #[32,30,30]
        x = F.relu(x)
        x = self.pool1(x)                #[32,15,15]

        x = self.conv2(x)                #[64,14,14]
        x = F.relu(x)
        x = self.pool2(x)                #[64,7,7]

        x = self.conv3(x)                #[64,6,6]
        x = F.relu(x)

        x = self.flatten(x)              #[64*6*6]
        x = self.linear1(x)              #input:64*6*6,output:64
        x = F.relu(x)
        x = self.linear2(x)              #input:64,output10
        return x

#定义selfLeNet网络   注意是Layer，不是layer
class selfLeNet(paddle.nn.Layer):
    def __init__(self):
        super(selfLeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=32,out_channels=64,kernel_size=5,stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2,stride=2)
        self.linear1 = paddle.nn.Linear(in_features=64*6*6,out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120,out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84,out_features=10)

    def forward(self,x):                                 #[3,32,32]
        x = self.conv1(x)                                #[32,32,32]
        x = F.relu(x)
        x = self.max_pool1(x)                            #[32,16,16]
        x = self.conv2(x)                                #[64,12,12]
        x = F.relu(x)
        x = self.max_pool1(x)                            #[64,6,6]
        x = paddle.flatten(x,start_axis=1,stop_axis=-1)  #[64*6*6]
        x = self.linear1(x)                              #input:64*6*6,output:120
        x = self.linear2(x)
        x = self.linear3(x)
        x = F.softmax(x)
        return x



if __name__ == '__main__':

    train_data = paddle.vision.datasets.cifar.Cifar10(mode='train')
    #model = paddle.Model(MyNet())
    input = InputSpec([None,3,32,32], 'float32', 'image')
    label = InputSpec([None,1], 'int64', 'label')
    model = paddle.Model(selfLeNet(),input,label)
    opt = paddle.optimizer.SGD(learning_rate=0.001,parameters=model.parameters())
    model.prepare(optimizer=opt,loss=paddle.nn.CrossEntropyLoss(),metrics=Accuracy())

    model.fit(train_data=train_data,epochs=1,batch_size=264,verbose=1)

    model.save('./ouput/cifar',training=True)
