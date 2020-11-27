import paddle
import matplotlib.pyplot as plt
import paddle.nn.functional as F
from paddle.metric import Accuracy

#定义数据
train_data = paddle.vision.datasets.MNIST(mode='train')   #train_data=[tuple,tuple],tuple = ((1,28,28),label)
# print(train_data.dtype)
# #print(len(train_data))
# print(train_data[0])
test_data = paddle.vision.datasets.MNIST(mode='test')

# train_data0, train_label_0 = train_data[0][0],train_data[0][1]    #train_data0的shape为(1,28,28) [[[  ]]]
# train_data0 = train_data0.reshape([28,28])
# plt.figure(figsize=(2,2))
# plt.imshow(train_data0, cmap=plt.cm.binary)
# print('train_data0 label is: ' + str(train_label_0))

#定义selfLeNet网络   注意是Layer，不是layer
class selfLeNet(paddle.nn.Layer):
    def __init__(self):
        super(selfLeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2,stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5,out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120,out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84,out_features=10)

    def forward(self,x):                            #out:  #[1,28,28]
        x = self.conv1(x)                                  #[6,28,28]
        x = F.relu(x)
        x = self.max_pool1(x)                              #[6,14,14]
        x = F.relu(x)
        x = self.conv2(x)                                  #[16,10,10]
        x = self.max_pool2(x)                              #[16,5,5]
        x = paddle.flatten(x,start_axis=1,stop_axis=-1)    #[16*5*5]
        x = self.linear1(x)                         #input:16*5*5,output:120
        x = self.linear2(x)                         #input:120,output:84
        x = self.linear3(x)                         #input:84,output:10
        return x

#初始化模型
model = paddle.Model(selfLeNet())
opt = paddle.optimizer.Adam(learning_rate=0.0001,parameters=model.parameters())   #parameters指定优化器优化的参数
model.prepare(optimizer=opt,loss=paddle.nn.CrossEntropyLoss(),metrics=Accuracy(topk=(1,2)))

#训练
model.fit(train_data=train_data,batch_size=64,epochs=3,verbose=1)

#预测
model.evaluate(eval_data=test_data,batch_size=64,verbose=1)