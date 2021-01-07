"""
对个图片处理，合成视频就 哈哈
# plt.imread/cv2.imread/PIL.Image.open的区别
# plt.imread和PIL.Image.open读入的都是RGB顺序，而opencv中cv2.imread读入的是BGR通道顺序 。cv2.imread会显示图片更蓝一些。
# imread  是array
# open    是一个对象
"""

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image
import utils.enhance_img as eni
import numpy as np

def Pic2Video():
    imgPath = "/home/chan/dataset/bili_video/"  # 读取图片路径
    videoPath = "/home/chan/dataset/test_up1.mp4"  # 保存视频路径

    images = os.listdir(imgPath)

    fps = 25  # 每秒25帧数

    # VideoWriter_fourcc为视频编解码器
    # ('I', '4', '2', '0') —>(.avi) 、
    # ('P', 'I', 'M', 'I')—>(.avi)、
    # ('X', 'V', 'I', 'D')—>(.avi)、
    # ('T', 'H', 'E', 'O')—>.ogv、
    # ('F', 'L', 'V', '1')—>.flv、
    # ('m', 'p', '4', 'v')—>.mp4
    fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')

    size = (600, 600)
    videoWriter = VideoWriter(videoPath, fourcc, fps, size)  #

    # 排序哦
    # 't{}.jpg'.format(str(im_name).zfill(4)) 拼接名称，达到名称排序

    for im_name in range(len(images)):
        frame = Image.open(imgPath + 't{}.jpg'.format(str(im_name+1).zfill(4)))  # 这里的路径只能是英文路径
        # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype = np.uint8), 1)  # 此句话的路径可以为中文路径

        frame = eni.enhance_Img(frame, 1.5, 1.5, 1.5, 1.5)

        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        frame = resize(frame, size, cv2.INTER_CUBIC)
        print('\rt{}.jpg'.format(str(im_name).zfill(4)), end="")
        videoWriter.write(frame)
    print("\n图片转视频结束！")
    videoWriter.release()
    cv2.destroyAllWindows()



def Video2Pic():
    videoPath = "/home/chan/dataset/聚众斗殴/视频/mda-ihmf80dupbkhagj2.mp4"  # 读取视频路径
    imgPath = "/home/chan/dataset/bili_video/t"  # 保存图片路径

    cap = cv2.VideoCapture(videoPath)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        if suc:
            cv2.imwrite(imgPath + (str(frame_count)+".jpg").zfill(8), frame)
        cv2.waitKey(1)
    cap.release()
    print("视频转图片结束！")

if __name__ == '__main__':
    #Video2Pic()
    Pic2Video()