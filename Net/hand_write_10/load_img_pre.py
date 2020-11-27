from Net.hand_write_10.hand_write_10_prediction import load_image
import paddle.fluid as fluid
import numpy as np
import PIL.Image as image
import matplotlib.pyplot as plt

"""
图片识别不太好 ，考虑重载load__image方法
"""

#加载图片
file_path = '../png/num_9.png'
tensor_img = load_image(file_path)

save_dirname = './hand_write_10.model'
exe = fluid.Executor(fluid.CPUPlace())

inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):
    # 使用 fluid.io.load_inference_model 获取 inference program desc,
    # feed_target_names 用于指定需要传入网络的变量名
    # fetch_targets 指定希望从网络中fetch出的变量名
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(
        save_dirname, exe, None, None)

    # 将feed构建成字典 {feed_target_name: feed_target_data}
    # 结果将包含一个与fetch_targets对应的数据列表
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: tensor_img},
                      fetch_list=fetch_targets)
    print(results)
    lab = np.argsort(results)

    # 打印 图片的预测结果
    # img=image.open(file_path)
    # plt.imshow(img)
    # plt.show()
    #最后一位最大数字对应的索引即是类别
    print("Inference result of image is: %d" % lab[0][0][-1])
    # print("Inference result of image is: %d" % np.argmax(results[0][0]))