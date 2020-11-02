#!/usr/bin/env python
# coding: utf-8
import os
import glob

# 忽略隐藏文件
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

path = os.getcwd()
print("[Work Path: {}]" .format(path))

list_dir = listdir_nohidden(path)

images = []
for image in list_dir:
    if image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg'):
        images.append(image)
print("Number of pictures: ", len(images))
print("---Start Compress---")
for i in images:
    image_name = os.path.basename(i)
    os.system('optimizt ' + image_name)

print("Done!")
