import matplotlib.pyplot as plt
import matplotlib.image as image
# img = image.imread('/home/chan/中山/新建文件夹/C0001A0ED.png')
# plt.imshow(img)
# plt.show()
# image.imsave('./png/a_002.png',img)

# def print_str():
#     print('1-2-3-4')



import os
# path = '/home/chan/中山/新建文件夹'
# img_list = os.listdir(path)
# i=0
# for file in img_list:
#     png_path = os.path.join(path,file)
#     #print(file)
#     img = cv2.imread(png_path)
#     new_path = '/home/chan/中山/png/a_00'+str(i)+'.png'
#     cv2.imwrite(new_path,img)
#     print(new_path)
#     i+=1

import cv2
# img = cv2.imread('/home/chan/中山/新建文件夹/C0001A0EF 快捷方式.png')
# cv2.imwrite('./png/a_002.png',img)
#
# path = '/home/chan/dataset/抽烟/xml'
# files = os.listdir(path)
# for file in files:
#     cur_file = os.path.join('/home/chan/dataset/抽烟/img',file.split('.')[0]+'.jpg')
#     new_file = os.path.join('/home/chan/dataset/抽烟/jpg',file.split('.')[0]+'.jpg')
#     os.rename(cur_file,new_file)