import os
import cv2


imgpath = '/home/chan/dataset/暴露垃圾'
files = os.listdir(imgpath)
i = 0
for file in files:
    file = os.path.join(imgpath,file)
    img = cv2.imread(file)
    img = cv2.resize(img, (600, 600))
    outpath = '/home/chan/dataset/暴露垃圾a/a_'+str(i)+'.jpg'
    cv2.imwrite(outpath,img)
    i+=1
print()