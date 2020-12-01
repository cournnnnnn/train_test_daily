import cv2
import os
# img = cv2.imread('/home/chan/中山/新建文件夹/C0001A0EF 快捷方式.png')
# cv2.imwrite('./png/a_002.png',img)

xml_path = '/home/chan/dataset/聚众斗殴/xml'
img_path ='/home/chan/dataset/聚众斗殴/img'
jpg_path = '/home/chan/dataset/聚众斗殴/jpg'


img_files = os.listdir(img_path)

#将img中的图片转换成jpg后缀
for img in img_files:
    fn,fd = img.split('.')
    if fd == 'jpg':
        pass
    else:
        old_file = os.path.join(img_path,img)
        old_file = cv2.imread(old_file)
        new_file = os.path.join(img_path,fn+'.jpg')
        cv2.imwrite(new_file,old_file)
        #os.rename(old_file,new_file)

#将有标签的img剪切到jpg文件中
files = os.listdir(xml_path)
for file in files:
    cur_file = os.path.join(img_path,file.split('.')[0]+'.jpg')
    new_file = os.path.join(jpg_path,file.split('.')[0]+'.jpg')
    os.rename(cur_file,new_file)