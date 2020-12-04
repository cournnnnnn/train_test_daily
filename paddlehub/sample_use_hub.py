import paddlehub as hub
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as image
import os


img_path = '../spider/download_img/9.jpg'
img_name,img_type = os.path.basename(img_path).split('.')

"""
人像扣图：deeplabv3p_xception65_humanseg
人体部位检测：ace2p
人脸检测：ultra_light_fast_generic_face_detector_1mb_640
关键点检测：human_pose_estimation_resnet50_mpii

图像分类：vgg、xception、shufflenetv2、se_resnet、resnet、resnet_vd、resnet_v2、pnasnet、mobilenet、inception_v4、Googlenet、efficientent、dpn、densent、darknet、alexnet
关键点检测：pose_resnet50_mpii、face_landmark_localization
目标检测：yolov3、ssd、Pyramidbox、faster_rcnn
图像生成： StyleProNNet、stgan、cyclegan、attgan
图像分割：deeplabv3、ace2p
视频分类：TSN、TSM、stnet、nonlocal


"""
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

