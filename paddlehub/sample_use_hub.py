import paddlehub as hub
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as image
import os


img_path = '../spider/download_img/9.jpg'
img_name,img_type = os.path.basename(img_path).split('.')

human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")
"""
images (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
paths (list[str]): 图片的路径；
batch_size (int): batch 的大小；
use_gpu (bool): 是否使用 GPU；
visualization (bool): 是否将识别结果保存为图片文件；
output_dir (str): 图片的保存路径。
"""
result = human_seg.segmentation(images=[cv2.imread(img_path)],
                                visualization=True,
                                output_dir='./output/')
#  return [{'save_path','data'}]
# or
# result = human_seg.segmentation(paths=['/PATH/TO/IMAGE'])
# img = image.imread('/home/chan/IdeaProjects/train_test_daily/spider/download_img/12.jpg')
plt.figure()
plt.imshow(np.array(result[0]['data']))
plt.axis('off')
#plt.savefig('./{}.png'.format(img_name))
plt.show()

