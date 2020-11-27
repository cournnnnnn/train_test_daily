import paddle
import paddle.distributed as dist
#from SelfDefineLoss import SoftmaxWithCrossEntropy
def first_model_fit_eva():

    #读取数据
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    val_dataset = paddle.vision.datasets.MNIST(mode='test')
    print(len(train_dataset))

    #定义网络
    mnist = paddle.nn.Sequential(
        paddle.nn.Flatten(),#28*28--->784)
        paddle.nn.Linear(784, 512),
        paddle.nn.ReLU(),
        paddle.nn.Dropout(0.2),#输入通道有20%的概率被至为0
        paddle.nn.Linear(512, 10)
    )

    # 预计模型结构生成模型实例，便于进行后续的配置、训练和验证
    model = paddle.Model(mnist)

    # 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
    model.prepare(paddle.optimizer.Adam(parameters=mnist.parameters()),
                  loss=
                  #SoftmaxWithCrossEntropy(),有问题
                  paddle.nn.CrossEntropyLoss(),
                  metrics=paddle.metric.Accuracy())

    # 开始模型训练
    model.fit(train_dataset,
              epochs=1,
              batch_size=32,
              verbose=1)
    print('-------------------------------------------------------------')
    #模型评估
    model.evaluate(val_dataset, verbose=1)



if __name__ == '__main__':

    first_model_fit_eva()


# #print-->model.parameters()
# a=mnist.state_dict()
# for key in a.keys():
#     print(key,a[key].shape)
#
# print()