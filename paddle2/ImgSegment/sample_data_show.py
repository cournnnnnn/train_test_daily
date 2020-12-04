import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddle2.ImgSegment.build_imgsegment_model import PetDataset


val_image_path = './images/'
label_images_path = './annotations/trimaps/'
val_dataset = PetDataset(val_image_path, label_images_path, mode='test')

image,label = val_dataset[3]
# 进行图片的展示
plt.figure()

plt.subplot(1,2,1),
plt.title('Train Image')
plt.imshow(np.array(image.transpose(1,2,0)).astype('uint8'))
plt.axis('off')

plt.subplot(1,2,2),
plt.title('Label')
plt.imshow(label[0].astype('uint8'), cmap='gray')
plt.axis('off')

plt.show()
