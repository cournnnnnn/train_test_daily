import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import os
from PIL import Image
from io import BytesIO
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def main(kw,output_dir,rol_n):

    option = webdriver.ChromeOptions()
    #规避浏览器检测自动控制
    option.add_experimental_option(
        'excludeSwitches',
        ['enable-automation'])
    #隐藏浏览器动作
    option.add_argument('headless') # 设置option

    brower = webdriver.Chrome('./chromedriver',options=option)
    brower.get('https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word={}'.format(kw))
    #ActionChains(driver).drag_and_drop_by_offset()#需要找滚动条element，目标位置，x,y的偏移
    for i in range(int(rol_n)):
        brower.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(0.5)

    html = brower.page_source
    brower.quit()    #获取到当前页码,浏览器关闭
    soup = BeautifulSoup(html,'html.parser')
    # print(soup.prettify())
    elements = soup.find_all(class_=['newfcImgli'])
    jpg_urls = []

    for ele in elements:
        url = ele.img.get('src')
        jpg_urls.append(url)

    elements2 = soup.find_all(class_=['imgitem'] )
    for ele in elements2:
        url = ele['data-objurl']
        jpg_urls.append(url)

    #print(len(jpg_urls))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    headers = {
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
    }
    #下载图片
    num_jpg = 0
    for url in jpg_urls:
        jpg = url.split('.')[-1]
        filename = '{}/{}.{}'.format(output_dir,num_jpg,jpg)
        img_type = ['png','jpg','jpeg']
        if jpg not in img_type:
            continue
        #请求可能无法响应
        try:
            resp_img = requests.get(url,headers=headers,timeout=(1,2)).content
        except:
            continue
        #print(url,'\t  start to download')

        # with open(filename,'wb') as f:
        #     f.write(resp_img)
        #响应返回可能不是图片
        try:
            img = Image.open(BytesIO(resp_img))
            img = img.resize((600, 600),Image.ANTIALIAS)
            img.save(filename)
        except:
            continue
        num_jpg+=1
        #覆盖打印的 ‘\r’不可或缺
        print('\r download:{:.2f} complte !!!'.format(num_jpg/len(jpg_urls)),end="")
    print()
    print('get url total:{} and download image total:{}'.format(len(jpg_urls),num_jpg))

if __name__ == '__main__':
    #搜索关键词
    kw = '公园'
    #保存路径
    output_dir = './download_img/'
    #滚动条拖动次数,与爬去图片数量相关
    rol_n=1
    main(kw,output_dir,rol_n)