import PIL.Image as image
import numpy as np
path ='./a_000.png'
#读取图片
img = image.open(path)
#img转array
img_array = np.array(img)
print('img_array: {}'.format(img_array))
#array转img
array_img = image.fromarray(np.uint8(img_array))
array_img.show()