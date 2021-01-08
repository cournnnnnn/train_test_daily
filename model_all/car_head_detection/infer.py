import os

import ffmpeg
import yaml

from PIL import Image
import cv2
import numpy as np
import paddle.fluid as fluid

import paddle
paddle.enable_static()

def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str/np.ndarray): path of image/ np.ndarray read by cv2
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_info['origin_shape'] = im.shape[:2]
        im_info['resize_shape'] = im.shape[:2]
    else:
        im = im_file
        im_info['origin_shape'] = im.shape[:2]
        im_info['resize_shape'] = im.shape[:2]
    return im, im_info

class Resize(object):
    """resize image by target_size and max_size
    Args:
        arch (str): model type
        target_size (int): the target size of image
        max_size (int): the max size of image
        use_cv2 (bool): whether us cv2
        image_shape (list): input shape of model
        interp (int): method of resize
    """

    def __init__(self,
                 arch,
                 target_size,
                 max_size,
                 use_cv2=True,
                 image_shape=None,
                 interp=cv2.INTER_LINEAR):
        self.target_size = target_size
        self.max_size = max_size
        self.image_shape = image_shape,
        self.arch = arch
        self.use_cv2 = use_cv2
        self.interp = interp
        self.scale_set = {'RCNN', 'RetinaNet'}

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im_channel = im.shape[2]
        im_scale_x, im_scale_y = self.generate_scale(im)
        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            resize_w = int(im_scale_x * float(im.shape[1]))
            resize_h = int(im_scale_y * float(im.shape[0]))
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)

        # padding im when image_shape fixed by infer_cfg.yml
        if self.max_size != 0 and self.image_shape is not None:
            padding_im = np.zeros(
                (self.max_size, self.max_size, im_channel), dtype=np.float32)
            im_h, im_w = im.shape[:2]
            padding_im[:im_h, :im_w, :] = im
            im = padding_im

        if self.arch in self.scale_set:
            im_info['scale'] = im_scale_x
        im_info['resize_shape'] = im.shape[:2]
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.max_size != 0 and self.arch in self.scale_set:
            im_size_min = np.min(origin_shape[0:2])
            im_size_max = np.max(origin_shape[0:2])
            im_scale = float(self.target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            im_scale_x = float(self.target_size) / float(origin_shape[1])
            im_scale_y = float(self.target_size) / float(origin_shape[0])
        return im_scale_x, im_scale_y

class Normalize(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        is_channel_first (bool): if True: image shape is CHW, else: HWC
    """

    def __init__(self, mean, std, is_scale=True, is_channel_first=False):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_channel_first:
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im, im_info

class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, to_bgr=False, channel_first=True):
        self.to_bgr = to_bgr
        self.channel_first = channel_first

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        if self.channel_first:
            im = im.transpose((2, 0, 1)).copy()
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im, im_info

class PadStride(object):
    """ padding image for model with FPN
    Args:
        stride (bool): model with FPN need image shape % stride == 0
    """

    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        coarsest_stride = self.coarsest_stride
        if coarsest_stride == 0:
            return im
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        im_info['resize_shape'] = padding_im.shape[1:]
        return padding_im, im_info

def create_inputs(im, im_info, model_arch='YOLO'):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        model_arch (str): model type
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = im
    origin_shape = list(im_info['origin_shape'])
    resize_shape = list(im_info['resize_shape'])
    scale = im_info['scale']
    if 'YOLO' in model_arch:
        im_size = np.array([origin_shape]).astype('int32')
        inputs['im_size'] = im_size
    elif 'RetinaNet' in model_arch:
        im_info = np.array([resize_shape + [scale]]).astype('float32')
        inputs['im_info'] = im_info
    elif 'RCNN' in model_arch:
        im_info = np.array([resize_shape + [scale]]).astype('float32')
        im_shape = np.array([origin_shape + [1.]]).astype('float32')
        inputs['im_info'] = im_info
        inputs['im_shape'] = im_shape
    return inputs

class Config():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """
    support_models = ['YOLO', 'SSD', 'RetinaNet', 'RCNN', 'Face']

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.use_python_inference = yml_conf['use_python_inference']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask_resolution = None
        if 'mask_resolution' in yml_conf:
            self.mask_resolution = yml_conf['mask_resolution']

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in self.support_models:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError(
            "Unsupported arch: {}, expect SSD, YOLO, RetinaNet, RCNN and Face".
                format(yml_conf['arch']))

# 目标检测消除多余框
def nms(boxes, threshold, method):
    # 非极大值抑制,去除重复多余框
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick

def load_predictor(model_dir,
                   run_mode='fluid',
                   batch_size=1,
                   use_gpu=False,
                   min_subgraph_size=3):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need use_gpu == True.
    """
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
                .format(run_mode, use_gpu))
    precision_map = {
        'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
        'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
        'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
    }
    config = fluid.core.AnalysisConfig(
        os.path.join(model_dir, '__model__'),
        os.path.join(model_dir, '__params__'))
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(100, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=run_mode == 'trt_int8')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor

def load_executor(model_dir, use_gpu=False):
    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=model_dir,
        executor=exe,
        model_filename='__model__',
        params_filename='__params__')
    return exe, program, fetch_targets

class Detector():
    """
    Args:
        model_dir (str): root path of __model__, __params__ and infer_cfg.yml
        use_gpu (bool): whether use gpu
    """

    def __init__(self,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):
        self.config = Config(model_dir)
        if self.config.use_python_inference:
            self.executor, self.program, self.fecth_targets = load_executor(
                model_dir, use_gpu=use_gpu)
        else:
            self.predictor = load_predictor(
                model_dir,
                run_mode=run_mode,
                min_subgraph_size=self.config.min_subgraph_size,
                use_gpu=use_gpu)
        self.preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            op_type = op_info.pop('type')
            if op_type == 'Resize':
                op_info['arch'] = self.config.arch
            self.preprocess_ops.append(eval(op_type)(**op_info))

    def preprocess(self, im):
        # process image by preprocess_ops
        im_info = {
            'scale': 1.,
            'origin_shape': None,
            'resize_shape': None,
        }
        im, im_info = decode_image(im, im_info)
        for operator in self.preprocess_ops:
            im, im_info = operator(im, im_info)
        im = np.array((im,)).astype('float32')
        inputs = create_inputs(im, im_info, self.config.arch)
        return inputs, im_info

    def postprocess(self, np_boxes, np_masks, im_info, threshold=0.5):
        # postprocess output of predictor
        results = {}
        if self.config.arch in ['SSD', 'Face']:
            w, h = im_info['origin_shape']
            np_boxes[:, 2] *= h
            np_boxes[:, 3] *= w
            np_boxes[:, 4] *= h
            np_boxes[:, 5] *= w
        expect_boxes = np_boxes[:, 1] > threshold
        np_boxes = np_boxes[expect_boxes, :]

        results['boxes'] = np_boxes
        if np_masks is not None:
            np_masks = np_masks[expect_boxes, :, :, :]
            results['masks'] = np_masks
        return results

    def predict(self, image, threshold=0.5):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape:[N, class_num, mask_resolution, mask_resolution]
        '''
        inputs, im_info = self.preprocess(image)
        np_boxes, np_masks = None, None
        if self.config.use_python_inference:
            outs = self.executor.run(self.program,
                                     feed=inputs,
                                     fetch_list=self.fecth_targets,
                                     return_numpy=False)
            if len(outs)<1:
                results = {'boxes': []}
                print('未检测出!!!')
                return results
            np_boxes = np.array(outs[0])
            if self.config.mask_resolution is not None:
                np_masks = np.array(outs[1])
        else:
            input_names = self.predictor.get_input_names()
            for i in range(len(inputs)):
                input_tensor = self.predictor.get_input_tensor(input_names[i])
                input_tensor.copy_from_cpu(inputs[input_names[i]])
            self.predictor.zero_copy_run()

            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_tensor(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if self.config.mask_resolution is not None:
                masks_tensor = self.predictor.get_output_tensor(output_names[1])
                np_masks = masks_tensor.copy_to_cpu()
        if len(np_boxes[0]) < 4:
            results = {'boxes': []}
            print('未检测出!!!')
            return results
        results = self.postprocess(
            np_boxes, np_masks, im_info, threshold=threshold)
        return results

def predict_image(model_dir, img_list, threshold=0.7, is_nms=True, nms_threshold=0.8, nms_method='Min'):
    data = []
    infer_scope = fluid.core.Scope()
    with fluid.scope_guard(infer_scope):
        detector = Detector(
            model_dir, use_gpu=use_gpu, run_mode=run_mode)

        for img in img_list:
            results = detector.predict(img, threshold)

            # 消除重复框
            total_boxes = results['boxes']
            if is_nms and len(total_boxes) > 0:
                total_boxes = total_boxes[:, [0, 2, 3, 4, 5, 1]]
                pick = nms(total_boxes[:, 1:], nms_threshold, nms_method)
                total_boxes = total_boxes[pick, :]
            if not is_nms and len(total_boxes) > 0:
                total_boxes = total_boxes[:, [0, 2, 3, 4, 5, 1]]
            data.append(total_boxes)
    return data

# def predict_video(model_dir, video_file, output_dir, threshold=0.7):
#     detector = Detector(
#         model_dir, use_gpu=use_gpu, run_mode=run_mode)
#     capture = cv2.VideoCapture(video_file)
#     fps = 30
#     width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_name = os.path.split(video_file)[-1]
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     out_path = os.path.join(output_dir, video_name)
#     writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
#     index = 1
# 
#     rotat = get_rotation(video_file)
#     rotat = int(rotat)
#     while (1):
#         ret, frame = capture.read()
#         if rotat == 90:
#             frame = cv2.rotate(frame, 0)
#         elif rotat == 180:
#             frame = cv2.rotate(frame, 1)
#         elif rotat == 270:
#             frame = cv2.rotate(frame, 2)
#         if not ret:
#             break
#         print('detect frame:%d' % (index))
#         index += 1
#         results = detector.predict(frame, threshold)
#         im = visualize_box_mask(
#             frame,
#             results,
#             detector.config.labels,
#             mask_resolution=detector.config.mask_resolution)
#         im = np.array(im)
#         writer.write(im)
#     writer.release()

# 从视频中获取旋转角度

def get_rotation(vider_path):
    try:
        vider_info_json = ffmpeg.probe(vider_path)
        rotate = vider_info_json["streams"][0]['tags']['rotate']
    except:
        print('动作活体检测中视频角度未获取到')
        rotate = 0
    return rotate

def run(img_list=[], confs_threshold=0.3):
    result = predict_image(model_path, img_list, confs_threshold, is_nms=False)
    return result

def dis_top2top(p1,p2):
    import math
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

# mode of running(fluid/trt_fp32/trt_fp16/trt_int8)
run_mode = 'fluid'
use_gpu = False
model_path = "./model"
label_list = ['head']

if __name__ == '__main__':
    import time
    img_path = "/home/chan/IdeaProjects/train_test_daily/jpg/car_head_694.jpg"
    img = cv2.imread(img_path)
    img_filename,_ = os.path.splitext(img_path.split("/")[-1])

    (img_w, img_h, _) = img.shape
    sum_area = img_w*img_h

    top_1 = (0, 0)
    top_2 = (img_w, 0)
    top_3 = (0, img_h)
    top_4 = (img_w, img_h)

    a = time.time()
    total_boxes = run([img], 0.3)[0]

    result = []
    for item in total_boxes:
        name = label_list[int(item[0])]
        # [label,x1,y1,x2,y2,score]
        result.append([name] + list(item[1:-1].astype('int')) + list(item[-1:]))
    print(time.time() - a)

    for num, dt in enumerate(result):
        # 画矩形框
        cv2.rectangle(img=img, pt1=(dt[1], dt[2]),
                      pt2=(dt[3], dt[4]), color=(0, 0, 255), thickness=1)
        # 写上类别名称
        cv2.putText(img, dt[0], (dt[1]+20, dt[2]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        # 写上识别类别的概率
        # cv2.putText(img, "%.3f" % dt[5], (dt[3], dt[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #             (0, 255, 255), 1)

        print("类别: " + dt[0])

    cv2.imwrite('{}p.jpg'.format(img_filename), img)

    for num, dt in enumerate(result):
        x = int((dt[1]+dt[2])/2)
        y = int((dt[2]+dt[4])/2)
        area = int((dt[3]-dt[1])*(dt[4]-dt[2])/sum_area*100)
        print("area : {}%".format(area))

        d1, d2, d3, d4 = dis_top2top(top_1, (x, y)), dis_top2top(top_2, (x, y)), dis_top2top(top_3, (x, y)), dis_top2top(top_4, (x, y))
        print("distinct top_1: {:.2f}\n"
              "distinct top_2: {:.2f}\n"
              "distinct top_3: {:.2f}\n"
              "distinct top_4: {:.2f}\n".format(d1, d2, d3, d4))
