import paddle.fluid as fluid
import numpy
#准备输入的X-Y
train_x = numpy.array([[1],[2],[3],[4]],dtype='float32')
true_y = numpy.array([[2],[4],[6],[8]],dtype='float32')

#定义输入
data_x = fluid.data(name='x',shape=[None,1],dtype='float32')
data_y = fluid.data(name='y',shape=[None,1],dtype='float32')

#定义net-全连接
y_predict = fluid.layers.fc(input=data_x,size=1,act=None)

#定义损失函数
loss = fluid.layers.square_error_cost(input=y_predict,label=data_y)
mean_loss = fluid.layers.mean(loss)

#定义优化参数
opt = fluid.optimizer.SGD(learning_rate=0.01)
opt.minimize(mean_loss)

#迭代训练
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


for i in range(30):
    outs = exe.run(program=fluid.default_main_program(),
    feed={'x':train_x,'y':true_y},
            fetch_list=[y_predict,mean_loss])

    print(outs[1])
