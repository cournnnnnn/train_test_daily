import paddle.fluid as fluid
#tensor_data
#data = fluid.data(name='x',shape=[2,2],dtype='int32')
"""
name: "x"
type {
    type: LOD_TENSOR
    lod_tensor {
        tensor {
        data_type: INT32
        dims: 2
        dims: 2
    }
        lod_level: 0
}
}
persistable: false
need_check_feed: true
"""
#print(data)
#build_content
data_content = fluid.layers.fill_constant(shape=[2,3],value=1,dtype='int32')
data_content = fluid.layers.Print(data_content,message='print data_content: ')
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
exe.run()
