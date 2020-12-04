import paddlehub as hub
from paddlehub.dataset.base_cv_dataset import BaseCVDataset  #加载图像类自定义数据集，仅需要继承基类BaseCVDatast，修改数据集存放地址即可
import paddle



paddle.enable_static()


class DemoDataset(BaseCVDataset):
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "/home/chan/IdeaProjects/train_test_daily/dataset/eye"  #dataset_dir为数据集实际路径，需要填写全路径
        super(DemoDataset, self).__init__(
            base_path=self.dataset_dir,
            train_list_file="./txt/train.txt",
            validate_list_file="./txt/val.txt",
            test_list_file="./txt/test.txt",
            #predict_file="predict_list.txt",     #如果还有预测数据（没有文本类别），可以将预测数据存放在predict_list.txt文件
            label_list_file="./txt/label_list.txt",
            # label_list=["数据集所有类别"]         #如果数据集类别较少，可以不用定义label_list.txt，可以选择定义label_list=["数据集所有类别"]
        )

#1. 准备数据与预模型
dataset = DemoDataset()
module = hub.Module(name="resnet_v2_50_imagenet")    #paddle.enable_static()
data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),   #预期桃子图片经过reader处理后的图像宽度
    image_height=module.get_expected_image_height(), #预期桃子图片经过reader处理后的图像高度
    images_mean=module.get_pretrained_images_mean(), #进行桃子图片标准化处理时所减均值。默认为None
    images_std=module.get_pretrained_images_std(),   #进行桃子图片标准化处理时所除标准差。默认为None
    dataset=dataset)

#2. 设置 Finetune API
config = hub.RunConfig(
    use_cuda=False,                                                #是否使用GPU训练，默认为False；
    num_epoch=100,                                                  #Fine-tune的轮数；
    checkpoint_dir="./output/cv_finetune_turtorial_demo",         #模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
    batch_size=32,                                                #训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
    eval_interval=50,                                             #模型评估的间隔，默认每100个step评估一次验证集；
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy())     #Fine-tune优化策略；

#3. 获取模型input/output/program，
input_dict, output_dict, program = module.context(trainable=True) #获取module的上下文信息包括输入、输出变量以及paddle program
img = input_dict["image"] #待传入图片格式
feature_map = output_dict["feature_map"] #从预训练模型的输出变量中找到最后一层特征图，提取最后一层的feature_map
feed_list = [img.name] #待传入的变量名字列表

#4. 新建task
task = hub.ImageClassifierTask(
    data_reader=data_reader,        #提供数据的Reader
    feed_list=feed_list,            #待feed变量的名字列表
    feature=feature_map,            #输入的特征矩阵
    num_classes=dataset.num_labels, #分类任务的类别数量
    config=config)


#5. finetune and eval
#run_states = task.finetune_and_eval() #通过众多finetune API中的finetune_and_eval接口，可以边训练，边打印结果

#6. predict
#auto Load the best model from ./output/cv_finetune_turtorial_demo/best_model
data = ["/home/chan/IdeaProjects/train_test_daily/dataset/eye/P0172.jpg"]      #传入一张测试M2类别的桃子照片
result = task.predict(data=data,return_result=True) #使用PaddleHub提供的API实现一键结果预测，return_result默认结果是False
print(result)