import paddle

class SoftmaxWithCrossEntropy(paddle.nn.Layer):
    def __init__(self):
        super(SoftmaxWithCrossEntropy, self).__init__()

    def forward(self, input, label):
        loss = paddle.fluid.layers.square_error_cost(input=input,label=label)
        mean_loss = paddle.fluid.layers.mean(loss)
        return mean_loss

SoftmaxWithCrossEntropy(1,1)