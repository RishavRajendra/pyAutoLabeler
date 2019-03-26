#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Rishav Rajendra"
__license__ = "MIT"
__status__ = "Development"

import os
import sys
import logging
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment
from pytorch.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from pytorch.utils.misc import Timer
import cv2

import warnings
warnings.filterwarnings('ignore')

# If user does not provide folder path, throw exception
if len(sys.argv) < 5:
    print('Usage: python3 img_to_xml.py <path_to_images> <path_to_annotations> <model_path> <label_path>')
    sys.exit(0)

# Assuming folder with images is in the same directory as test
img_path = sys.argv[1].rstrip('/')
img_dir_name = img_path.split('/')[-1]
path_to_image_dir = "{}/{}".format(os.getcwd(), img_path)

# Path where all the xml files are saved
xml_path = sys.argv[2].rstrip('/')
# Path to trained model
model_path = sys.argv[3]
# Path to label map
label_path = sys.argv[4]

# Check if folders are present
assert os.path.isdir(path_to_image_dir), \
'Image path provided does not exist'
assert os.path.isdir(xml_path), 'XML save folder does not exist'

# Enters spaces into the XML file to make it more readable
def enterSpacesInXML(elem):
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
	# Iterate through all the files in the image folder provided
	for image in os.listdir(path_to_image_dir):
		if image.endswith(".jpg") or image.endswith(".png"):

			img = cv2.imread('/'.join((path_to_image_dir, image)))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			# Header of the XML file
			header = Element('annotation')

			# Name of the dir image resides in
			dir_name = SubElement(header, 'folder')
			dir_name.text = '{}'.format(img_dir_name)

			# Name of the current image
			img_name = SubElement(header, 'filename')
			img_name.text = '{}'.format(image)

			# Path to current image
			path = SubElement(header, 'path')
			path.text = '{}/{}'.format(path_to_image_dir, image)

			# TODO: Source database. Placeholder for now
			source = SubElement(header, 'source')
			database = SubElement(source, 'database')
			database.text = 'Unknown'

			# height, width and depth of current image
			img_height, img_width, img_depth = img.shape

			# Append image width, height and depth info of image to XML
			size = SubElement(header, 'size')
			width = SubElement(size, 'width')
			width.text = '{}'.format(img_width)
			height = SubElement(size, 'height')
			height.text = '{}'.format(img_height)
			depth = SubElement(size, 'depth')
			depth.text = '{}'.format(img_depth)

			# TODO: Image segmentation. Place holder: 0
			segmentation = SubElement(header, 'segmented')
			segmentation.text = '0'
			
			# Read label names from voc _files
			class_names = [name.strip() for name in open(label_path).readlines()]
			
			# Initialize net
			net = create_vgg_ssd(len(class_names), is_test=True)
			net.load(model_path)
			
			# Predict objects
			predictor = create_vgg_ssd_predictor(net, candidate_size=200)
			# Change 0.8 to change probability threshold
			boxes, labels, probs = predictor.predict(img, 10, 0.8)
			
			# Go through all the detected objects
			for i in range(boxes.size(0)):
				# box data
				box = boxes[i, :]
				label = '{}'.format(class_names[labels[i]])

				objectDetected = SubElement(header, 'object')
				
				# append object label in XML
				objectName = SubElement(objectDetected, 'name')
				objectName.text = '{}'.format(label)
				
				# append object pose in XML
				objectPose = SubElement(objectDetected, 'pose')
				objectPose.text = 'Unspecified'

				# assuming label will not be truncated or marked difficult
				objectTruncated = SubElement(objectDetected, 'truncated')
				objectTruncated.text = '0'
				objectDifficult = SubElement(objectDetected, 'difficult')
				objectDifficult.text = '0'

				# append box coords to XML
				objectBoxCoords = SubElement(objectDetected, 'bndbox')
				xmin = SubElement(objectBoxCoords, 'xmin')
				xmin.text = '{}'.format(int(box[0]))
				ymin = SubElement(objectBoxCoords, 'ymin')
				ymin.text = '{}'.format(int(box[1]))
				xmax = SubElement(objectBoxCoords, 'xmax')
				xmax.text = '{}'.format(int(box[2]))
				ymax = SubElement(objectBoxCoords, 'ymax')
				ymax.text = '{}'.format(int(box[3]))
			
			# Write XML file to path
			f = open('{}/{}.xml'.format(xml_path,image.split('.')[0]),'w')
			f.write(enterSpacesInXML(header))

if __name__ == '__main__':
	main()
