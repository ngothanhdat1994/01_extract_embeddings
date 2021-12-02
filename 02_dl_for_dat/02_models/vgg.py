
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate

from util import *

def vgg(nF_aud, nT_aud, nC_aud, nClass, chanDim=-1):
   #----------------------------------------- input define
   input_shape_aud = (nF_aud, nT_aud, nC_aud)
   inputs_aud = Input(shape=input_shape_aud)

   #---------------------------------------- network architecture
   #---Audio stream
   #BLOCK 01:
   x= dob_conv2d(inputs_aud,
                 conv2d_filters=32, conv2d_kernel=(3,3), 
                 is_bn=True,
                 act_type='relu',
                 is_pool=True, pool_type='AP', pool_size=(1,2), pool_strides=(1,2), pool_padding='valid',
                 drop_rate=0.2
                )
   #BLOCK 02: 
   x= dob_conv2d(x,
                 conv2d_filters=64, conv2d_kernel=(3,3), 
                 is_bn=True,
                 act_type='relu',
                 is_pool=True, pool_type='AP',pool_size=(1,2), pool_strides=(1,2), pool_padding='valid',
                 drop_rate=0.25
                )

   #BLOCK 03:
   x= dob_conv2d(x,
                 conv2d_filters=128, conv2d_kernel=(3,3), 
                 is_bn=True,
                 act_type='relu',
                 is_pool=False, pool_type='AP',pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
                 drop_rate=0.3
                )

   #BLOCK 04:
   x= dob_conv2d(x,
                 conv2d_filters=128, conv2d_kernel=(3,3), 
                 is_bn=True,
                 act_type='relu',
                 is_pool=True, pool_type='AP',pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
                 drop_rate=0.3
                )
   #BLOCK 05:
   x= dob_conv2d(x,
                 conv2d_filters=256, conv2d_kernel=(3,3), 
                 is_bn=True,
                 act_type='relu',
                 is_pool=False, pool_type='AP',pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
                 drop_rate=0.35
                )

   #BLOCK 06:
   x= dob_conv2d(x,
                 conv2d_filters=256, conv2d_kernel=(3,3), 
                 is_bn=True,
                 act_type='relu',
                 is_pool=True, pool_type='GAP',
                 drop_rate=0.35
                )

   #---concatenate aud and img streams
   #BLOCK 07:
   x = l_dense(x, den_unit=1024)
   x = l_act(x, 'relu')
   x = l_drop(x, 0.4)

   #BLOCK 08:
   x = l_dense(x, den_unit=256)
   x = l_act(x, 'relu')
   x = l_drop(x, 0.4)

   #BLOCK 09:  
   x = l_dense(x, den_unit=2)
   x = l_act(x, 'softmax', "output")

   #---output
   name='vgg'
   output = Model(inputs=inputs_aud, outputs=x, name=name)

   return output
   

