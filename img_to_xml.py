#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Rishav Rajendra"
__license__ = "MIT"
__status__ = "Development"

import os
import sys
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment
import cv2

import warnings
warnings.filterwarnings('ignore')

# If user does not provide folder name, Throw exception
if len(sys.argv) < 2:
    print('Usage: python3 test.py <folder_with_images>')
    sys.exit(0)

# Assuming folder with images is in the same directory as test
img_dir_name = sys.argv[1].rstrip('/')
path_to_image_dir = "{}/{}".format(os.getcwd(), img_dir_name)

# Make sure the folder is in current directory
assert os.path.isdir(path_to_image_dir), \
'Folder provided does not exist. Make sure folder is in current directory.'

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
	# Iterate through all the files in the image folder provided
	for image in os.listdir(path_to_image_dir):
		if image.endswith(".jpg") or image.endswith(".png"):
			# Open current image and convert to RGB
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

			print(prettify(header))

if __name__ == '__main__':
	main()