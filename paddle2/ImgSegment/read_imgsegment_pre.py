import paddle
from paddle2.ImgSegment.build_imgsegment_model import PetDataset,PetNet,SoftmaxWithCrossEntropy
from matplotlib import pyplot as plt
import numpy as np


n_image = 3#显示几幅图
val_image_path = './images/'#预测图片地址
label_images_path = './annotations/trimaps/'#图片label地址
pre_dataset = PetDataset(val_image_path, label_images_path, mode='prediction')

model = paddle.Model(PetNet(37))
model.load('./model/final/12')
opt = paddle.optimizer.RMSProp(learning_rate=0.001,
                               rho=0.9,
                               momentum=0.0,
                               epsilon=1e-07,
                               centered=False,
                               parameters=model.parameters())
model.prepare(optimizer=opt,loss=SoftmaxWithCrossEntropy())

result = model.predict(pre_dataset)         #N*C*H*L,C是class的C

img_index=0
plt.figure()
for i in range(len(result[0])):

    image,label = pre_dataset[i]

    if i ==n_image:
        break

    plt.subplot(n_image,3,img_index+1)
    plt.imshow(np.array(image).transpose(1,2,0).astype(np.uint8))
    plt.title('image')
    plt.axis('off')

    plt.subplot(n_image,3,img_index+2)
    plt.imshow(np.array(label[0]).astype(np.uint8),cmap='gray')
    plt.title('label')
    plt.axis('off')


    #最后绘制预测结果不太明白，这个mask
    #result.shape
    data = result[0][i][0].transpose((1, 2, 0))       #shape(160*160*37)
    mask = np.argmax(data, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    plt.subplot(n_image,3,img_index+3)
    #plt.imshow(mask.astype('uint8'), cmap='gray')
    plt.imshow(np.squeeze(mask, axis=2).astype('uint8'), cmap='gray')
    plt.title('Predict')
    plt.axis("off")
    img_index+=3

plt.show()