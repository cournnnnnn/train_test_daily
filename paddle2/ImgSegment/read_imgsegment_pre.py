from paddle2.ImgSegment import build_imgsegment_model
import paddle
from matplotlib import pyplot as plt
import numpy as np



if __name__ == "__main__":

    num_classes = 37
    #验证集是从训练集中，切分出来的，so~~
    train_images_path = ''
    label_images_path = ''
    # 验证数据集
    val_dataset = build_imgsegment_model.PetDataset(train_images_path, label_images_path, mode='test')

    model = paddle.Model(build_imgsegment_model.PetNet(num_classes))
    model.load('./model/final')

    opt = paddle.optimizer.RMSProp(learning_rate=0.0001,parameters=model.parameters())
    loss = build_imgsegment_model.SoftmaxWithCrossEntropy()
    model.prepare(optimizer=opt,loss=loss)
    # #继续训练的
    # model.fit()
    predict_results = model.predict(val_dataset)       #预测结果为NCHW，模型最后一层是conv2d,C ==num_classes


    print(len(predict_results))
    plt.figure(figsize=(10, 10))

    i = 0
    mask_idx = 0

    for data in val_dataset:
        if i > 8:
            break
        plt.subplot(3, 3, i + 1)
        plt.imshow(data[0].transpose((1, 2, 0)).astype('uint8'))
        plt.title('Input Image')
        plt.axis("off")

        plt.subplot(3, 3, i + 2)
        plt.imshow(np.squeeze(data[1], axis=0).astype('uint8'), cmap='gray')
        plt.title('Label')
        plt.axis("off")

        # 模型只有一个输出，所以我们通过predict_results[0]来取出1000个预测的结果
        # 映射原始图片的index来取出预测结果，提取mask进行展示
        data = predict_results[0][mask_idx][0].transpose((1, 2, 0))  #mask_idx的预测结果
        mask = np.argmax(data, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        plt.subplot(3, 3, i + 3)
        plt.imshow(np.squeeze(mask, axis=2).astype('uint8'), cmap='gray')
        plt.title('Predict')
        plt.axis("off")
        i += 3
        mask_idx += 1

    plt.show()

