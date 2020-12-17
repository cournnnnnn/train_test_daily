"""
主要图片增强
附加：matplotpylib显示中文字

"""

from matplotlib import pyplot as plt
import PIL.Image as image
from PIL import ImageEnhance
import matplotlib as mb

def enhance_Img(img,brightness,color,contrast,sharpness):

    # =================================================================
    # "亮度增亮"
    enh_bri = ImageEnhance.Brightness(img)
    #brightness = 1.8
    image_brightened = enh_bri.enhance(brightness)
    # =================================================================
    # "色度增强"
    enh_col = ImageEnhance.Color(image_brightened)
    #color = 1.5
    image_colored = enh_col.enhance(color)
    # =================================================================
    # "对比度增强"
    enh_con = ImageEnhance.Contrast(image_colored)
    #contrast = 2
    image_contrasted = enh_con.enhance(contrast)
    # =================================================================
    # "锐度增强"
    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    #sharpness = 1.2
    image_sharped = enh_sha.enhance(sharpness)
    # =================================================================



    return image_sharped


zhfont = mb.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

if __name__ == '__main__':

    baseimg_path = '../dataset/test_data/img/a-102.jpg'
    img = image.open(baseimg_path)
    img2 = enhance_Img(img, 1.5, 1.5, 1.5, 1.5)
# =================================================================
    title = "new 新图"
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.title("old 原图", fontproperties=zhfont)
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(img2)
    plt.title(title, fontproperties=zhfont)
    plt.axis("off")

    plt.show()