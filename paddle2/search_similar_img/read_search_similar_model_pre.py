import paddle
from paddle2.search_similar_img import build_search_similar_img_model
import numpy as np
from collections import defaultdict
import random

path = "./search_similar/model2"
model = paddle.jit.load(path)

num_classes = 10
height_width = 32

x_test,y_test = build_search_similar_img_model.trans_dataset('test')

x_test_t = paddle.to_tensor(x_test).astype('float32')
test_images_embeddings = model(x_test_t)
#在test集中亮亮比较相似度
similarities_matrix = paddle.matmul(test_images_embeddings, test_images_embeddings, transpose_y=True)
#获取每个test图片的相似度排名索引
indicies = paddle.argsort(similarities_matrix, descending=True)
indicies = indicies.numpy()

#相似度最高的前10
near_neighbours_per_example = 10
examples = np.empty(
    (
        num_classes,
        near_neighbours_per_example + 1,
        3,
        height_width,
        height_width,
    ),
    dtype=np.float32,
)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)

for row_idx in range(num_classes):
    examples_for_class = class_idx_to_test_idxs[row_idx]
    anchor_idx = random.choice(examples_for_class)

    examples[row_idx, 0] = x_test[anchor_idx] #搜索IMG
    anchor_near_neighbours = indicies[anchor_idx][1:near_neighbours_per_example+1] #搜索到的IMGS
    for col_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx + 1] = x_test[nn_idx]

build_search_similar_img_model.show_collage(examples).show()