import paddle.fluid as fluid
import paddle
import matplotlib.pyplot as plt
import PIL.Image as image
import numpy as np

#softmax_regression
def softmax_regression():
    img = fluid.data(name='img',shape=[None,1,28,28],dtype='float32')#img_1*28*28
    prediction = fluid.layers.fc(input=img,size=10,act='softmax')
    return prediction

#多层感知机
def multilayer_perceptorn():
    img = fluid.data(name='img',shape=[None,1,28,28],dtype='float32')
    hidden1 = fluid.layers.fc(input=img,size=200,act='relu')
    hidden2 = fluid.layers.fc(input=hidden1,size=60,act='relu')
    prediction = fluid.layers.fc(input=hidden2,size=10,act='softmax')
    return prediction

#卷积——池化
def conv_pool(input,conv_num,conv_size,pool_size,pool_step,act='relu'):
    conv_out = fluid.layers.conv2d(  #图片是2维的
        input=input,num_filters=conv_num,filter_size=conv_size,act=act
    )
    pool_out = fluid.layers.pool2d(
        input=conv_out,pool_size=pool_size,pool_stride=pool_step
    )
    return pool_out

def convolutional_nerural_network():
    img = fluid.data(name='img',shape=[None,1,28,28],dtype='float32')
    conv_pool_1 = conv_pool(
        input=img,conv_num=50,conv_size=5,act='relu',
        pool_size=2,pool_step=2
    )
    # 归一化处理
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)

    conv_pool_2 = conv_pool(
        input=conv_pool_1,conv_num=50,conv_size=5,act='relu',
        pool_size=2,pool_step=2
    )
    predicton = fluid.layers.fc(input=conv_pool_2,size=10,act='softmax')
    return predicton

def train_program(net_num):
    label = fluid.data(name='label',shape=[None,1],dtype='int64')
    net = {
        1:softmax_regression(),
        2:multilayer_perceptorn(),
        3:convolutional_nerural_network()
    }
    # prediction = softmax_regression()
    # prediction = multilayer_perceptorn()
    # prediction = convolutional_nerural_network()
    prediction = net.get(net_num)

    loss = fluid.layers.cross_entropy(input=prediction,label=label)
    avg_loss = fluid.layers.mean(loss)

    acc = fluid.layers.accuracy(input=prediction,label=label)
    return prediction,[avg_loss,acc]

def testprogram(exe,test_program,test_feed,test_reader):
    test_acc_list=[]
    test_loss_list=[]
    for test in test_reader():
        test_avg_loss,test_acc = exe.run(program=test_program,
                feed=test_feed.feed(test),
                fetch_list=[avg_loss,acc])
        test_acc_list.append(test_acc)
        test_loss_list.append(test_avg_loss)
    avg_losses = np.array(test_loss_list).mean()
    avg_accs = np.array(test_acc_list).mean()
    return avg_losses,avg_accs

def load_image(file):
    # 读取图片文件，并将它转成灰度图
    im = image.open(file).convert('L')
    # 将输入图片调整为 28*28 的高质量图
    im = im.resize((28, 28), image.ANTIALIAS)
    # 将图片转换为numpy
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    # 对数据作归一化处理
    im = im / 255.0 * 2.0 - 1.0
    return im



if __name__ == '__main__':


    #读取数据
    batch_size = 60
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(),batch_size=batch_size
    )
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(),batch_size=batch_size
    )
    #初始化网络
    place = fluid.CPUPlace()
    prediction,[avg_loss,acc] = train_program(1)
    feeder = fluid.DataFeeder(feed_list=['img','label'],place=place )

    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()
    test_program = fluid.default_main_program().clone(for_test=True)

    optSGD = fluid.optimizer.SGD(learning_rate=0.001)
    optSGD.minimize(avg_loss)

    exe = fluid.Executor(place)
    exe.run(program=start_program)
    exe_test = fluid.Executor(place)

    #定义训练次数
    pass_num = 10

    lists = []
    #迭代训练
    for i in range(pass_num):
        for train in train_reader():
            results = exe.run(program=main_program,
                              feed=feeder.feed(train),
                              fetch_list=[prediction,avg_loss,acc])
        print('train_i:{}---avg_train_loss:{}---acc:{}'.format(i,results[1],results[2]))

        #训练集上测试
        avg_test_loss,avg_test_acc = testprogram(exe=exe_test,
                                                 test_program=test_program,
                                                  test_feed=feeder,
                                                  test_reader=test_reader)
        lists.append((avg_test_loss,avg_test_acc))
        best = sorted(lists,key=lambda list:float(list[0]))[0]
        print('avg_test_loss:{}---acc:{}'.format(avg_test_loss,avg_test_acc))
        print('best---\n test_loss:{}\n test_acc:{}'.format(best[0],best[1]))

    #保存模型
    save_model_path = './hand_write_10.model'
    fluid.io.save_inference_model(dirname=save_model_path,
                                  feeded_var_names=['img'],
                                  target_vars=[prediction],
                                  executor=exe)