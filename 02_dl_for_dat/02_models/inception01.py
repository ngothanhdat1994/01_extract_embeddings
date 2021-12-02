
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate

from util import *

def inception01(nF_aud, nT_aud, nC_aud, nClass, chanDim=-1):
   #----------------------------------------- input define
   input_shape_aud = (nF_aud, nT_aud, nC_aud)
   inputs_aud = Input(shape=input_shape_aud)

   #---------------------------------------- network architecture
   #---Audio stream
   #BLOCK 01:
   x = l_batch_norm(inputs_aud, bn_moment=0.9, bn_dim=-1)
   x = l_inc_01(x, 8, 32, 16, 8)
   x = l_batch_norm(x, bn_moment=0.9, bn_dim=-1)
   x = l_max_pool2d(x, pool_size=(1,4), pool_strides=(4,4), pool_padding='valid')
   x = l_drop(x, drop_rate=0.1)

   #BLOCK 02:
   x = l_batch_norm(x, bn_moment=0.9, bn_dim=-1)
   x = l_inc_01(x, 16, 64, 32, 16)
   x = l_batch_norm(x, bn_moment=0.9, bn_dim=-1)
   x = l_max_pool2d(x, pool_size=(1,4), pool_strides=(2,2), pool_padding='valid')
   x = l_drop(x, drop_rate=0.15)

   #BLOCK 03:
   x = l_batch_norm(x, bn_moment=0.9, bn_dim=-1)
   x = l_inc_01(x, 32, 128, 64, 32)
   x = l_batch_norm(x, bn_moment=0.9, bn_dim=-1)
   x = l_max_pool2d(x, pool_size=(1,4), pool_strides=(2,2), pool_padding='valid')
   x = l_drop(x, drop_rate=0.2)

   #BLOCK 04:
   x = l_batch_norm(x, bn_moment=0.9, bn_dim=-1)
   x = l_inc_01(x, 64, 256, 128, 64)
   x = l_batch_norm(x, bn_moment=0.9, bn_dim=-1)
   x = l_glo_max_pool2d(x)
   x = l_drop(x, drop_rate=0.25)

   #---concatenate aud and img streams
   #BLOCK 07:
   x = l_dense(x, den_unit=1024)
   x = l_act(x, 'relu')
   x = l_drop(x, 0.3)

   #BLOCK 08:
   x = l_dense(x, den_unit=1024)
   x = l_act(x, 'relu')
   x = l_drop(x, 0.3)

   #BLOCK 09:  
   x = l_dense(x, den_unit=2)
   x = l_act(x, 'softmax', "output")

   #---output
   name='inception01'
   output = Model(inputs=inputs_aud, outputs=x, name=name)

   return output
   

