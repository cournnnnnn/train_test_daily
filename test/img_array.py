import PIL.Image as image
import numpy as np
path ='/home/chan/IdeaProjects/train_test_daily/spider/download_img/6.jpg'
#读取图片
img = image.open(path)
#img转array
img_array = np.array(img)
print('img_array: {}'.format(img_array))
#array转img
array_img = image.fromarray(np.uint8(img_array))
array_img.show()