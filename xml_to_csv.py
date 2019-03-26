import os
import sys
import glob
import pandas as pd
import xml.etree.ElementTree as ET

if len(sys.argv) < 2:
    print('Usage: python xml_to_csv.py <dataset_path>')
    sys.exit(0)

directory_path = sys.argv[1].rstrip('/')

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    if not os.path.isdir('data'):
        os.mkdir('data')
    for directory in ['train', 'test']:
    	xml_df = xml_to_csv(os.path.join(directory_path, 'images/{}/Annotations'.format(directory)))
    	xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
    	print('Successfully converted xml to csv.')


main()
