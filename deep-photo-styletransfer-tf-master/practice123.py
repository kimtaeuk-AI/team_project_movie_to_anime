import cv2
import numpy as np

# a = cv2.imread('C:/Study/deep-photo-styletransfer-tf-master/taeuk_input/2_2.png')
# a= cv2.resize(a, (173,300))
# a.reshape((1,) + a.shape)
# print(a.shape)
# np.expand_dims(a, axis=0)

# print(a.shape)


# height,width, channel = a.shape
# print(height, width, channel)
import tensorflow as tf
output = 100

loss_affine = 0.0
output_t = output / 255.
for Vc in tf.unstack(output_t, axis=-1):
    Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
    loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.sparse_tensor_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

print (loss_affine)