import paddle

class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()

        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = paddle.nn.Linear(in_features=13,out_features=1)

    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.fc(inputs)
        return x

train_dataset = paddle.batch(paddle.dataset.uci_housing.train(),batch_size=32)
test_dataset = paddle.batch(paddle.dataset.uci_housing.test(),batch_size=32)

# for step, data in enumerate(train_dataset()):
#     print(step,',',data)


model = paddle.Model(Regressor())
opt = paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())
model.prepare(opt,paddle.nn.CrossEntropyLoss(),paddle.metric.Accuracy())
#有问题,train_dataset不是迭代，train_dataset()迭代了又参数不对
#model.fit(train_dataset,epochs=3,batch_size=32,verbose=1)