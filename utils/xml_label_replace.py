xml_files = '/media/yt/data/dataset/voc/kiechen/xml'

import xml.etree.ElementTree as ET
import os
for xml in os.listdir(xml_files):
    xml = os.path.join(xml_files,xml)
    doc=ET.parse(xml)
    root = doc.getroot()
    objects = root.findall('object')
    filename,jpg_type = os.path.splitext(root.find('filename').text)
    for object in objects:
        name = object.find('name').text
        if name == 'back':
            object.find('name').text = 'person'

    doc.write('/media/yt/data/dataset/voc/kiechen/xml_test/{}.xml'.format(filename))
    print('xml_test/{}.xml write complite'.format(filename))