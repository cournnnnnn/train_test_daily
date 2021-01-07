import os
from PIL import Image

def util(key):
    path = '/home/chan/dataset/{}'.format(key)
    files = os.listdir(path)
    i = 0
    mode_list = ['1', 'L', 'I', 'F', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr']
    save_path = '/home/chan/dataset/{}/img/'.format(key)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in files:
        f = os.path.join(path, file)
        img = Image.open(f)
        img = img.convert('RGB')
        # img = img.resize((608, 608), Image.ANTIALIAS)
        img.save(save_path+'card_{:04d}.jpg'.format(i))  #空位补0

        i+=1
        print(save_path+'card_{:04d}.jpg'.format(i))

if __name__ == '__main__':
    key = '行驶证'
    util(key)