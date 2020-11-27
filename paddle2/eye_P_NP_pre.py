import paddle
from paddle2 import eye_P_NP_fit_save
import cv2
import numpy as np
from paddle.static import InputSpec
import os

def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = np.array(img).reshape(-1,1,3,224,224).astype(np.float32)

    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img

#继续训练
def continue_to_fit():

    train_dataset = eye_P_NP_fit_save.MyDataset(mode='train')
    lenet = eye_P_NP_fit_save.LeNet(2)
    model = paddle.Model(lenet)
    model.load('./output/final')
    opt = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters())
    model.prepare(opt,paddle.nn.CrossEntropyLoss(),metrics=paddle.metric.Accuracy())
    model.fit(train_dataset,epochs=2,batch_size=10,verbose=1)

def continue_to_pre():
    imgpath = '/home/chan/dataset/eye/PALM-Training400/PALM-Training400/P0031.jpg'
    img = cv2.imread(imgpath)
    img = transform_img(img)
    print(img.shape) #(3, 224, 224)
    list_result = {0:'NP',1:'P'}
    lenet = eye_P_NP_fit_save.LeNet(2)
    input = InputSpec([None,3,32,32],dtype='float32',name='image')
    label = InputSpec([None,1],dtype='int64',name='label')
    model = paddle.Model(lenet,input,label)

    opt = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters())
    model.load('./output/final')
    model.prepare(opt,paddle.nn.CrossEntropyLoss(),metrics=paddle.metric.Accuracy())

    result = model.predict(paddle.to_tensor(img))

    #result = model.predict(paddle.to_tensor(np.array([[img]])))
    #第一个索引是图片索引  result：  [(array([[-1.8550125 ,  0.33674115]], dtype=float32),)]
    #第二个索引是模型输出值   [[-1.8550125 ,  0.33674115]]
    # [-1.8550125 ,  0.33674115]
    #print(result)
    print('img: {} \n---predict result: {}'.format(os.path.basename(imgpath),list_result[np.argmax(result[0])]))

if __name__ == '__main__':
    #continue_to_fit()
    continue_to_pre()







