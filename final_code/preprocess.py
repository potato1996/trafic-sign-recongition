# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from skimage import io, color, exposure, transform
from copy import deepcopy

# Increase the contrast
def hist_norm(img):
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    return img
def rotate_90(img):
    return transform.rotate(img, 270.0)
def rotate_180(img):
    return transform.rotate(img, 180.0)
def rotate_270(img):
    return transform.rotate(img, 90.0)
def flip_vert(img):
    res = deepcopy(img)
    return res[:, ::-1, :]
def flip_horiz(img):
    res = deepcopy(img)
    return res[::-1, :, :]


# Off-line Augument - Apply hist_norm, rotate and flipping
# Also ensure that each class has at least 'count' of images
def aug_img_set(count, source_dir, aug_dir):
    aug_imgs = []
    for class_id in range(0, 43): aug_imgs += [[]]
    vert_flip_class_id = [11, 12, 13, 15, 17, 18, 22, 26, 30, 32, 35, 41]
    
    print('Before augment:')
    # load all images
    for class_id in range(0, 43):
        class_path = os.path.join(source_dir, '{:05d}'.format(class_id))
        class_img_paths = glob.glob(os.path.join(class_path, '*.ppm'))
        print('Class ' + str(class_id) + ' :' + str(len(class_img_paths)) + ' images')
        for img_path in class_img_paths:
            img = io.imread(img_path)
            
            # Get a image with better contrast
            img = hist_norm(img)
            
            aug_imgs[class_id].append(img)
        
            if class_id in vert_flip_class_id:
                aug_imgs[class_id].append(flip_vert(img))     
        
            if class_id == 15:
                aug_imgs[15].append(rotate_90(img))
                aug_imgs[15].append(rotate_180(img))
                aug_imgs[15].append(rotate_270(img))
                aug_imgs[15].append(flip_horiz(img))

            if class_id == 17:
                aug_imgs[17].append(rotate_180(img))
                aug_imgs[17].append(flip_horiz(img))

            if class_id == 19:
                aug_imgs[20].append(flip_vert(img))

            if class_id == 20:
                aug_imgs[19].append(flip_vert(img))

            if class_id == 33:
                aug_imgs[34].append(flip_vert(img))

            if class_id == 34:
                aug_imgs[33].append(flip_vert(img))

            if class_id == 36:
                aug_imgs[37].append(flip_vert(img))

            if class_id == 37:
                aug_imgs[36].append(flip_vert(img))

            if class_id == 38:
                aug_imgs[39].append(flip_vert(img))

            if class_id == 39:
                aug_imgs[38].append(flip_vert(img))
            
    # Ensure each class has at least 'count' images - basically trying to balance the classes
    # This in reality is **NOT** just duplicate the images - we have online augumentation
    # in the training codes, thus the duplicated images are effectively different during training
    print('After augment: ')
    for i in range(0, 43):
        curr_count = len(aug_imgs[i])
        dup_time = count // curr_count + 1
        aug_imgs[i] = aug_imgs[i] * dup_time
        print('Class ' + str(i) + ' :' + str(len(aug_imgs[i])) + ' images')
    
    # save everything back                            
    if not os.path.isdir(aug_dir):
        print(aug_dir + ' not found, expanding it')
        os.mkdir(aug_dir)
    for class_id in range(0, 43):
        class_path = os.path.join(aug_dir, '{:05d}'.format(class_id))
        if not os.path.isdir(class_path):
            os.mkdir(class_path)
        for i, img in enumerate(aug_imgs[class_id]):
            img_path = os.path.join(class_path, '{:05d}'.format(i) + '.png')
            io.imsave(img_path, img)


# We also need to pre-process the test set
def sharp_img(source_dir, dst_dir):
    img_paths = glob.glob(os.path.join(source_dir, '*.ppm'))
    if not os.path.isdir(dst_dir):
        print(dst_dir + ' not found, expanding it')
        os.mkdir(dst_dir)
    for img_path in img_paths:
            img = io.imread(img_path)
            img = hist_norm(img)
            save_path = os.path.join(dst_dir, os.path.basename(img_path))
            io.imsave(save_path, img)

from data import initialize_data

initialize_data('data') # extracts the zip files, makes a validation set

# In[5]:
aug_img_set(4000, 'data/train_images', 'data/train_aug_images')


# In[6]:
aug_img_set(100,  'data/val_images'  , 'data/val_aug_images')


# In[7]:
sharp_img('data/test_images', 'data/test_sharp_images')

