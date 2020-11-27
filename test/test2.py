from visualdl import LogWriter
import time
# # # 在`./log/scalar_test/train`路径下建立日志文件
# with LogWriter(logdir="./log/scalar_test/train") as writer:
#     for i in range(100):
#         # 使用scalar组件记录一个标量数据
#         time.sleep(1)
#         writer.add_scalar(tag="acc", step=i+1, value=2**i)

import numpy as np
from PIL import Image
from visualdl import LogWriter


def random_crop(img):
    """获取图片的随机 100x100 分片
    """
    img = Image.open(img)
    w, h = img.size
    random_w = np.random.randint(0, w - 100)
    random_h = np.random.randint(0, h - 100)
    r = img.crop((random_w, random_h, random_w + 100, random_h + 100))
    return np.asarray(r)


if __name__ == '__main__':
    # 初始化一个记录器
    with LogWriter(logdir="./log/image_test/train") as writer:
        for step in range(20):
            # 添加一个图片数据
            writer.add_image(tag="eye",
                             img=random_crop("./png/人像扣图.png"),
                             step=step)
