import paddle
import numpy as np
from paddle.io import Dataset
class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()

        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = paddle.nn.Linear(in_features=13,out_features=1)

    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.fc(inputs)
        return x

class bostonData(Dataset):

    def __init__(self, mode='train'):

        super(bostonData, self).__init__()

        self.data = []

        if mode == 'train':
            train_dataset = paddle.dataset.uci_housing.train()
            for train in train_dataset():
                train_x,train_y = train
                self.data.append([np.array(train_x).astype('float32'),np.array(train_y).astype('float32')])
            # self.data = [
            #     ['traindata1', 'label1'],
            # ]
        else:
            test_dataset = paddle.dataset.uci_housing.test()
            for test in test_dataset():
                test_x,test_y = test
                self.data.append([np.array(test_x).astype('float32'),np.array(test_y).astype('float32')])

    def __getitem__(self, index):

        data = self.data[index][0]
        label = self.data[index][1]

        return data, label

    def __len__(self):

        return len(self.data)

#自定义加载
train_dataset = bostonData(mode='train')
test_dataset = bostonData(mode='test')
# #不知道看第一遍的时候怎么看漏的
# train_dataset = paddle.text.datasets.UCIHousing(mode='train')
# eval_dataset = paddle.text.datasets.UCIHousing(mode='test')

model = paddle.Model(Regressor())
opt = paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())
model.prepare(opt,paddle.nn.MSELoss())


model.fit(train_data=train_dataset,eval_data=test_dataset,epochs=3,batch_size=32,verbose=1)