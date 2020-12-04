import numpy as np
import paddle
from paddle.nn import functional as F
from paddle.static import InputSpec
from paddle2.search_similar_img.build_search_similar_img_model import reader_creator,MyNet

# load
path = "./search_similar/model"
model = paddle.jit.load(path)

# loaded_model.eval()
# x = paddle.randn([1, IMAGE_SIZE], 'float32')
# pred = loaded_model(x)

# fine-tune
model.train()
num_classes = 10
pairs_train_reader = reader_creator(100)  #100*2*10*3*32*32
inverse_temperature = paddle.to_tensor(np.array([1.0/0.2], dtype='float32'))
epoch_num = 200

newmodel = MyNet()
newmodel.set_state_dict(paddle.load('/home/chan/IdeaProjects/train_test_daily/paddle2/search_similar/model'))
opt = paddle.optimizer.Adam(learning_rate=0.0001,
                            parameters=model.parameters())

for epoch in range(epoch_num):
    for batch_id, data in enumerate(pairs_train_reader()):
        anchors_data, positives_data = data[0], data[1]
        #anchors,positives-->10*3*32*32
        anchors = paddle.to_tensor(anchors_data)
        positives = paddle.to_tensor(positives_data)
        #anchor_embeddings,positive_embeddings--->10*8
        anchor_embeddings = newmodel(anchors)
        positive_embeddings = newmodel(positives)
        #matmul:[10*8]*[8*10]--->[10*10]
        similarities = paddle.matmul(anchor_embeddings, positive_embeddings, transpose_y=True)
        similarities = paddle.multiply(similarities, inverse_temperature)

        sparse_labels = paddle.arange(0, num_classes, dtype='int64')
        """
        cross_entropy Parameters:
                input (Tensor): Input tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
                label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        loss = F.cross_entropy(similarities, sparse_labels)

        if batch_id % 500 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
        loss.backward()
        opt.step()
        opt.clear_grad()

# 保存模型&参数
path = "./search_similar/model2"
paddle.jit.save(newmodel,path,input_spec=[InputSpec(shape=[None, 3,32,32], dtype='float32')])