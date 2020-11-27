


"""
save(training = True)
load(model) to continue fit
"""

"""   
start_to_continue fit...
len_data_set: 50000
Epoch 1/1
step 250/250 [==============================] - loss: 2.3912 - acc: 0.1120 - 276ms/step  
"""
from paddle.static import InputSpec
import paddle
from paddle2 import cifar10_pre
print('start_to_continue fit...')
train_data = paddle.vision.datasets.cifar.Cifar10(mode='train')
print('len_data_set: {}'.format(len(train_data)))

input = InputSpec([None,3,32,32], 'float32', 'image')
label = InputSpec([None,1], 'int64', 'label')

model = paddle.Model(cifar10_pre.selfLeNet(),input,label)
model.load('./ouput/cifar')

opt = paddle.optimizer.SGD(learning_rate=0.001,parameters=model.parameters())
model.prepare(opt,paddle.nn.CrossEntropyLoss(),paddle.metric.Accuracy())

model.fit(train_data,batch_size=264,epochs=1,verbose=1,shuffle=True)
model.save('./ouput/fit2/cifar',training=False)
