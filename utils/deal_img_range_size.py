import os
import cv2
path = '/home/chan/dataset/打架斗殴'

files = os.listdir(path)
i = 0
for file in files:
    curpath = os.path.join(path,file)
    img = cv2.imread(curpath)
    img = cv2.resize(img,(600,600))
    newpath = os.path.join(path,'a_{}.jpg'.format(i))
    cv2.imwrite(newpath,img)
    #os.rename(curpath,newpath)
    i+=1