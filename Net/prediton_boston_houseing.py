import argparse
import paddle.fluid as fluid
import paddle

def parse_arg():
    parse = argparse.ArgumentParser(description='here to describle the argparse')
    parse.add_argument('--enable_ce',type=bool,default=True)
    parse.add_argument('--iscpu',type=bool,default=True)
    arg = parse.parse_args()
    return arg


if __name__ == '__main__':
    batch_size=20
    arg=parse_arg()
    if arg.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(),
            batch_size=batch_size
        )
        test_reader = paddle.batch(
            paddle.dataset.uci_housing.test(),
            batch_size=batch_size
        )
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.train(),buf_size=300
            ),
            batch_size=batch_size
        )
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.test(),buf_size=300
            ),
            batch_size=batch_size
        )

    #定义数据
    x = fluid.data(name='x',shape=[None,13],dtype='float32')
    y = fluid.data(name='y',shape=[None,1],dtype='float32')

    #定义网络
    prediction = fluid.layers.fc(input=x,size=1,act=None)
    loss = fluid.layers.square_error_cost(input=prediction,label=y)
    avg_loss = fluid.layers.mean(loss)

    start_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()
    test_program = main_program.clone(for_test=True)


    #优化参数
    SGDopt = fluid.optimizer.SGD(learning_rate=0.001)
    SGDopt.minimize(avg_loss)

    place = fluid.CPUPlace() if arg.iscpu else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    #主循环
    feeder = fluid.DataFeeder(feed_list=[x,y],place=place)
    exe.run(program=start_program)

    n=0
    for i in range(20):
        for data_train in train_reader():
            avg_loss_value, = exe.run(program=main_program,feed=feeder.feed(data_train),fetch_list=[avg_loss])
            #每10批次打印一次损失值
            # if n%10 == 0:
            #    print('{}-step:{}-cost:{}'.format('train',n,avg_loss_value[0]))

            #每100批次打印一次测试
            # if n%100 == 0:
            #     for data_test in test_reader():
            #         result, = exe.run(program=test_program,feed=feeder.feed(data_test),
            #                           fetch_list=[avg_loss])
                    #print('result:{}'.format(result))

            n+=1
        print('echo: {}----{}'.format(i,avg_loss_value[0]))

    if arg.enable_ce and i == 19:
        print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
 #       print("kpis\ttest_cost\t%f" % result[0])