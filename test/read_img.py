
# """matplotlib"""
# import matplotlib.pyplot as plt
# import matplotlib.image as image
# img = image.imread('/home/chan/IdeaProjects/train_test_daily/spider/download_img/4.jpg')
# plt.imshow(img)
# plt.show()
# #image.imsave('./png/a_002.png',img)
# #plt.savefig('testblueline.jpg')


# """cv2 """
# import cv2
# import matplotlib.pyplot as plt
# img2 = cv2.imread('/home/chan/IdeaProjects/train_test_daily/spider/download_img/4.jpg')
# #OpenCV是以BGR模式读入图片，如果想要正常显示图片，则需要改成RGB格式
# img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)#格式转换，
# plt.imshow(img2)
# plt.show()
# #cv2.imwrite('./png/a_002.png',img)


# """PIL"""
# import PIL
# import matplotlib.pyplot as plt
# img1 = PIL.Image.open('/home/chan/IdeaProjects/train_test_daily/spider/download_img/11.jpg')
# img2 = PIL.Image.open('/home/chan/IdeaProjects/train_test_daily/paddlehub/output/ndarray_time=1606903981598882.png')
# #plt.figure()
# plt.subplot(1,2,1)
# plt.axis('off')
# plt.imshow(img1)
# plt.subplot(1,2,2)
# plt.imshow(img2)
# plt.axis('off')
# #plt.savefig('./test1.png')
# plt.show()
# #img1.save('./png/a_002.png')
