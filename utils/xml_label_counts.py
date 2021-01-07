xml_files = '/home/chan/dataset/行驶证/xml'

import xml.etree.ElementTree as ET
import os
import collections

labels_count_dict = collections.defaultdict()

for xml in os.listdir(xml_files):
    xml = os.path.join(xml_files,xml)
    doc = ET.parse(xml)
    root = doc.getroot()
    objects = root.findall('object')
    filename,jpg_type = os.path.splitext(root.find('filename').text)
    for object in objects:
        name = object.find('name').text
        if name not in labels_count_dict.keys():
            labels_count_dict[name] = 1
        else:
            labels_count_dict[name]+=1


labels_list = labels_count_dict.keys()

for k in labels_count_dict.keys():
    print("{}:{}".format(k,labels_count_dict[k]))

print("len: {}".format(len(labels_list)))
print(labels_list)
