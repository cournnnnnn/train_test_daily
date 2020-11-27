import paddle.fluid as fluid
import numpy as np

x = fluid.layers.data(name='x',shape=[1],dtype='float32',lod_level=1)
y = fluid.layers.data(name='y',shape=[1],dtype='float32',lod_level=2)
out = fluid.layers.sequence_expand(x=x,y=y,ref_level=1)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

x_ = fluid.create_lod_tensor(np.array([[8],[9],[10]]).astype('float32'),[[2,1]],place)
y_ = fluid.create_lod_tensor(np.array([[1],[2],[3],[4],[5]]).astype('float32'),[[2],[3,2]],place)

result = exe.run(fluid.default_main_program(),
        feed={'x':x_,'y':y_},
        fetch_list=[out],return_numpy=False)

print(np.array(result[0]))