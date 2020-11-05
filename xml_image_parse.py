# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:50:17 2020
@author: Adam
Quick n dirty: parses images from Pascal VOC 2012 image set via XML
"""
# XML files:
# describes location of several objects in the image
# parse file, output a separate image for each object
# each new image goes in a folder with the name of the object
# resize images to 224x224

from PIL import Image
import xml.etree.ElementTree as ET
import os

# loop and crop each object in each image from 2007-2012 image sets
for i in range(9950):
    try:
        tree = ET.parse(
            'C:/Users/Adam/Desktop/595/Week 5/Reduce overfitting/PascalVOC2012/VOC2012/Annotations/2007_00' + str(
                i).zfill(4) + '.xml')
        root = tree.getroot()

        # parse image and crop to a new image file for each object present
        for node in tree.findall('object'):
            image = Image.open('PascalVOC2012/VOC2012/JPEGImages/2007_00' + str(i).zfill(4) + '.jpg')
            # get object name, create folder
            path = 'PascalVOC2012/' + node[0].text
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            a = node.findall("bndbox")
            xmin = int(a[0].findall("xmin")[0].text)
            ymin = int(a[0].findall("ymin")[0].text)
            xmax = int(a[0].findall("xmax")[0].text)
            ymax = int(a[0].findall("ymax")[0].text)
            cropped = image.crop((xmin, ymin, xmax, ymax))
            cropped = cropped.resize([224, 224], Image.LANCZOS)
            cropped.save(path + '/' + '2007' + str(i) + str(node[0].text) + '.jpg')

    except FileNotFoundError as e:
        print('#' + str(i) + ' not found.')

for i in range(8774):
    try:
        tree = ET.parse(
            'C:/Users/Adam/Desktop/595/Week 5/Reduce overfitting/PascalVOC2012/VOC2012/Annotations/2008_00' + str(
                i).zfill(4) + '.xml')
        root = tree.getroot()

        # parse image and crop to a new image file for each object present
        for node in tree.findall('object'):
            image = Image.open('PascalVOC2012/VOC2012/JPEGImages/2008_00' + str(i).zfill(4) + '.jpg')
            # get object name, create folder
            path = 'PascalVOC2012/' + node[0].text
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            a = node.findall("bndbox")
            xmin = int(a[0].findall("xmin")[0].text)
            ymin = int(a[0].findall("ymin")[0].text)
            xmax = int(a[0].findall("xmax")[0].text)
            ymax = int(a[0].findall("ymax")[0].text)
            cropped = image.crop((xmin, ymin, xmax, ymax))
            cropped = cropped.resize([224, 224], Image.LANCZOS)
            cropped.save(path + '/' + '2008' + str(i) + str(node[0].text) + '.jpg')

    except FileNotFoundError as e:
        print('#' + str(i) + ' not found.')

for i in range(5312):
    try:
        tree = ET.parse(
            'C:/Users/Adam/Desktop/595/Week 5/Reduce overfitting/PascalVOC2012/VOC2012/Annotations/2009_00' + str(
                i).zfill(4) + '.xml')
        root = tree.getroot()

        # parse image and crop to a new image file for each object present
        for node in tree.findall('object'):
            image = Image.open('PascalVOC2012/VOC2012/JPEGImages/2009_00' + str(i).zfill(4) + '.jpg')
            # get object name, create folder
            path = 'PascalVOC2012/' + node[0].text
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            a = node.findall("bndbox")
            xmin = int(a[0].findall("xmin")[0].text)
            ymin = int(a[0].findall("ymin")[0].text)
            xmax = int(a[0].findall("xmax")[0].text)
            ymax = int(a[0].findall("ymax")[0].text)
            cropped = image.crop((xmin, ymin, xmax, ymax))
            cropped = cropped.resize([224, 224], Image.LANCZOS)
            cropped.save(path + '/' + '2009' + str(i) + str(node[0].text) + '.jpg')

    except FileNotFoundError as e:
        print('#' + str(i) + ' not found.')

for i in range(6995):
    try:
        tree = ET.parse(
            'C:/Users/Adam/Desktop/595/Week 5/Reduce overfitting/PascalVOC2012/VOC2012/Annotations/2010_00' + str(
                i).zfill(4) + '.xml')
        root = tree.getroot()

        # parse image and crop to a new image file for each object present
        for node in tree.findall('object'):
            image = Image.open('PascalVOC2012/VOC2012/JPEGImages/2010_00' + str(i).zfill(4) + '.jpg')
            # get object name, create folder
            path = 'PascalVOC2012/' + node[0].text
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            a = node.findall("bndbox")
            xmin = int(a[0].findall("xmin")[0].text)
            ymin = int(a[0].findall("ymin")[0].text)
            xmax = int(a[0].findall("xmax")[0].text)
            ymax = int(a[0].findall("ymax")[0].text)
            cropped = image.crop((xmin, ymin, xmax, ymax))
            cropped = cropped.resize([224, 224], Image.LANCZOS)
            cropped.save(path + '/' + '2010' + str(i) + str(node[0].text) + '.jpg')

    except FileNotFoundError as e:
        print('#' + str(i) + ' not found.')

for i in range(7215):
    try:
        tree = ET.parse(
            'C:/Users/Adam/Desktop/595/Week 5/Reduce overfitting/PascalVOC2012/VOC2012/Annotations/2011_00' + str(
                i).zfill(4) + '.xml')
        root = tree.getroot()

        # parse image and crop to a new image file for each object present
        for node in tree.findall('object'):
            image = Image.open('PascalVOC2012/VOC2012/JPEGImages/2011_00' + str(i).zfill(4) + '.jpg')
            # get object name, create folder
            path = 'PascalVOC2012/' + node[0].text
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            a = node.findall("bndbox")
            xmin = round(float((a[0].findall("xmin")[0].text)))
            ymin = round(float((a[0].findall("ymin")[0].text)))
            xmax = round(float((a[0].findall("xmax")[0].text)))
            ymax = round(float((a[0].findall("ymax")[0].text)))
            cropped = image.crop((xmin, ymin, xmax, ymax))
            cropped = cropped.resize([224, 224], Image.LANCZOS)
            cropped.save(path + '/' + '2011' + str(i) + str(node[0].text) + '.jpg')

    except FileNotFoundError as e:
        print('#' + str(i) + ' not found.')

for i in range(4332):
    try:
        tree = ET.parse(
            'C:/Users/Adam/Desktop/595/Week 5/Reduce overfitting/PascalVOC2012/VOC2012/Annotations/2012_00' + str(
                i).zfill(4) + '.xml')
        root = tree.getroot()

        # parse image and crop to a new image file for each object present
        for node in tree.findall('object'):
            image = Image.open('PascalVOC2012/VOC2012/JPEGImages/2012_00' + str(i).zfill(4) + '.jpg')
            # get object name, create folder
            path = 'PascalVOC2012/' + node[0].text
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            a = node.findall("bndbox")
            xmin = int(a[0].findall("xmin")[0].text)
            ymin = int(a[0].findall("ymin")[0].text)
            xmax = int(a[0].findall("xmax")[0].text)
            ymax = int(a[0].findall("ymax")[0].text)
            cropped = image.crop((xmin, ymin, xmax, ymax))
            cropped = cropped.resize([224, 224], Image.LANCZOS)
            cropped.save(path + '/' + '2012' + str(i) + str(node[0].text) + '.jpg')

    except FileNotFoundError as e:
        print('#' + str(i) + ' not found.')
