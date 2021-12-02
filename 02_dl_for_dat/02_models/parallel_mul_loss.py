
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate

import tensorflow as tf

from util import *


def parallel_mul_loss(nF_aud, nT_aud, nC_aud, nClass, chanDim=-1):
   #----------------------------------------- input define
   input_shape_aud = (nF_aud, nT_aud, nC_aud)
   input_01 = Input(shape=input_shape_aud, name='x1')
   input_02 = Input(shape=input_shape_aud, name='x2')

   #---------------------------------------- network architecture
   #---X1 stream
   #BLOCK 01:
   x1 = l_batch_norm(input_01, bn_moment=0.9, bn_dim=-1)
   x1 = l_inc_01(x1, 8, 32, 16, 8)
   x1 = l_batch_norm(x1, bn_moment=0.9, bn_dim=-1)
   x1 = l_max_pool2d(x1, pool_size=(4,4), pool_strides=(4,4), pool_padding='valid')
   x1 = l_drop(x1, drop_rate=0.1)

   #BLOCK 02:
   x1 = l_batch_norm(x1, bn_moment=0.9, bn_dim=-1)
   x1 = l_inc_01(x1, 16, 64, 32, 16)
   x1 = l_batch_norm(x1, bn_moment=0.9, bn_dim=-1)
   x1 = l_max_pool2d(x1, pool_size=(2,2), pool_strides=(2,2), pool_padding='valid')
   x1 = l_drop(x1, drop_rate=0.15)

   #BLOCK 03:
   x1 = l_batch_norm(x1, bn_moment=0.9, bn_dim=-1)
   x1 = l_inc_01(x1, 32, 128, 64, 32)
   x1 = l_batch_norm(x1, bn_moment=0.9, bn_dim=-1)
   x1 = l_max_pool2d(x1, pool_size=(2,2), pool_strides=(2,2), pool_padding='valid')
   x1 = l_drop(x1, drop_rate=0.2)

   #BLOCK 04:
   x1 = l_batch_norm(x1, bn_moment=0.9, bn_dim=-1)
   x1 = l_inc_01(x1, 64, 256, 128, 64)
   x1 = l_batch_norm(x1, bn_moment=0.9, bn_dim=-1)
   x1 = l_glo_max_pool2d(x1)
   x1 = l_drop(x1, drop_rate=0.25)
   x11 = l_dense(x1, 512)

   #---X2 stream
   #BLOCK 01:
   x2 = l_batch_norm(input_02, bn_moment=0.9, bn_dim=-1)
   x2 = l_inc_01(x2, 8, 32, 16, 8)
   x2 = l_batch_norm(x2, bn_moment=0.9, bn_dim=-1)
   x2 = l_max_pool2d(x2, pool_size=(4,4), pool_strides=(4,4), pool_padding='valid')
   x2 = l_drop(x2, drop_rate=0.1)

   #BLOCK 02:
   x2 = l_batch_norm(x2, bn_moment=0.9, bn_dim=-1)
   x2 = l_inc_01(x2, 16, 64, 32, 16)
   x2 = l_batch_norm(x2, bn_moment=0.9, bn_dim=-1)
   x2 = l_max_pool2d(x2, pool_size=(2,2), pool_strides=(2,2), pool_padding='valid')
   x2 = l_drop(x2, drop_rate=0.15)

   #BLOCK 03:
   x2 = l_batch_norm(x2, bn_moment=0.9, bn_dim=-1)
   x2 = l_inc_01(x2, 32, 128, 64, 32)
   x2 = l_batch_norm(x2, bn_moment=0.9, bn_dim=-1)
   x2 = l_max_pool2d(x2, pool_size=(2,2), pool_strides=(2,2), pool_padding='valid')
   x2 = l_drop(x2, drop_rate=0.2)

   #BLOCK 04:
   x2 = l_batch_norm(x2, bn_moment=0.9, bn_dim=-1)
   x2 = l_inc_01(x2, 64, 256, 128, 64)
   x2 = l_batch_norm(x2, bn_moment=0.9, bn_dim=-1)
   x2 = l_glo_max_pool2d(x2)
   x2 = l_drop(x2, drop_rate=0.25)
   x22 = l_dense(x2, 512)

   #---concatenate aud and img streams
   #BLOCK 05:
   x = tf.math.add(x11,x22)
   x = l_act(x, 'relu')
   x = l_dense(x, den_unit=2048)
   x = l_act(x, 'relu')
   x = l_drop(x, 0.3)

   #BLOCK 06:
   x = l_dense(x, den_unit=512)
   x = l_act(x, 'relu')
   x = l_drop(x, 0.3)

   #BLOCK 07:  
   x = l_dense(x, den_unit=4)
   x = l_act(x, 'softmax', "output_01")

   #BLOCK 08:  
   xb1 = l_dense(x1, den_unit=4)
   xb1 = l_act(xb1, 'softmax', "output_02")

   xb2 = l_dense(x2, den_unit=4)
   xb2 = l_act(xb2, 'softmax', "output_03")

   #---output
   name='inception01'
   output = Model(inputs=[input_01, input_02], outputs=[x, xb1, xb2], name=name)

   return output
   

