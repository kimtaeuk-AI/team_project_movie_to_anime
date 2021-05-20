from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from vgg19.vgg import Vgg19
from PIL import Image
import time
from closed_form_matting import getLaplacian
import math
from functools import partial
import copy
import os

# try:
#     xrange          # Python 2
# except NameError:
#     xrange = range  # Python 3

# VGG_MEAN = [103.939, 116.779, 123.68]

# def rgb2bgr(rgb, vgg_mean=True):
#     if vgg_mean :
#         return rgb[:,:,::-1] - VGG_MEAN
#     else:
#         return rgb[:,:,::-1]

# def bgr2rgb(bgr,vgg_mean=False):
#     if vgg_mean :
#         return bgr[:,:,::-1] + VGG_MEAN
#     else:
#         return rgb[:,:,::-1]

# def load_seg(content_seg_path, style_seg_path, content_shape, style_shape):
#     color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']
#     def _extract_mask(seg, color_str)
#         h,w,c = np.shape(seg)
#         if color_str == 'BLUE'
#             mask_r = (seg[:,:,0]<0.1).astype(np.unit8) # 2

#     content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR), dtype=np.float32)/255.0

#     color_content_masks = []
#     color_style_masks = []
#     for i in xrange(len(color_codes)):
#         color_content_masks.append(tf.expand_dims(tf.expand_dims))

# import tensorflow as tf
import numpy as np
# a=np.array([[[ 1,  2,  3],
#                   [ 4,  5,  6]],
#                  [[ 7,  8,  9],
#                   [10, 11, 12]]])
# print(a.shape)

# b = tf.transpose(a,[0,1,2])

# print(b.shape)

# c = tf.transpose(a,[0,2,1])

# print(c.shape)

# c = tf.transpose(a,[0,2,3])

# _, content_seg_height, content_seg_wdth,_ = content_segs[]

import cv2
from PIL import Image, ImageShow
# import matplotlib.pyplot as plt

# a = cv2.imread('C:/Study/deep-photo-styletransfer-tf-master/wonbin.jpg')
# cv2.imshow('a',a)
# cv2.waitKey()
# cv2.destroyAllWindows()


# im = Image.open('C:/Study/deep-photo-styletransfer-tf-master/out_iter_4000.png')
# plt.imshow(a[:,:,::-1]) 
# plt.show()
# plt.imshow(a) #bgr
# plt.show()

# VGG_MEAN = [103.939, 116.779, 123.68]
# mean_pixel = tf.constant(VGG_MEAN)
# print(mean_pixel)

# a= np.array([[1,1],[2,1]])
# b= np.array([[1,1],[2,1]])
# print(a)
# gram = tf.transpose(a, [1, 0])
# print(gram)
# gram = tf.matmul(a, b, transpose_b=True)
# print(gram.shape)

layer_structure_all = [layer.name for layer in vgg_var.get_all_layers()]

style_loss(layer_structure_all)