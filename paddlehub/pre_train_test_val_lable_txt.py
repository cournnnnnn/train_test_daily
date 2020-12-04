import os
import cv2
import random

base_path = '/home/chan/dataset/eye/PALM-Training400/PALM-Training400'

def get_list(path):

    save_data_path ="../dataset/eye"
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)

    jpg = 'jpg'
    files = os.listdir(path)
    img_labels = []
    for file in files:
        name,imgtype = os.path.basename(file).split('.')
        if name[0] == 'H':
            label = "0"
        elif name[0] == "N":
            label = "1"
        else:
            #"P"
            label = "2"

        f = os.path.join(base_path,file)
        if imgtype != 'jpg':
            img = cv2.imread(f)
            cv2.imwrite('{}/{}.jpg'.format(save_data_path,name),img)
        else:
            os.system('cp {} {}'.format(f,'{}/{}.jpg'.format(save_data_path,name)))

        img_labels.append(['./'+file,label])
    return img_labels

def get_txt_list(img_labels):
    random.shuffle(img_labels)
    filter_line = 0.8
    train_data = img_labels[:int(len(img_labels)*filter_line)]
    test_data = img_labels[int(len(img_labels)*filter_line):]
    val_data = img_labels[int(len(img_labels)*filter_line):]

    datas = {'train':train_data,'test':test_data,'val':val_data}

    for data in datas.keys():
        #目录是否存在
        save_data_path ='../dataset/eye/txt'
        if not os.path.exists(save_data_path):
            os.mkdir(save_data_path)
        #文件是否存在
        file_path = os.path.join(save_data_path,data+".txt")
        if os.path.isfile(file_path):
            os.remove(file_path)
        #空的文件里写入数据
        with open(os.path.join(save_data_path,data+".txt"),'w') as f:
            for train in datas[data]:
                f.write(train[0]+" "+train[1]+'\n')
        print("create {}.txt complite !!!".format(data))

def create_lable_list():

    file_path = '../dataset/eye/txt/{}.txt'.format("label_list")
    if os.path.isfile(file_path):
        os.remove(file_path)

    with open(file_path,'w') as f:
        f.write("H\n")
        f.write("N\n")
        f.write("P\n")

    print("create label_list.txt complite !!!")



if __name__ == '__main__':
    img_labels = get_list(base_path)
    get_txt_list(img_labels)
    create_lable_list()