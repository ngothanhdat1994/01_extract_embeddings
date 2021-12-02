#-------------------- Tensoflow packages
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
#---
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D   
from tensorflow.keras.layers import GlobalMaxPooling2D   
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Dense
#import keras.layers.concatenate
from tensorflow.keras.layers import Concatenate


#-------------------- Layer definition
def l_batch_norm(x, bn_moment=0.9, bn_dim=-1):
    return BatchNormalization(axis     = bn_dim,
                              momentum = bn_moment
                             )(x)

def l_conv2d(x, conv2d_filters, conv2d_kernel, conv2d_stride=(1,1), conv2d_padding='same', conv2d_dilation_rate=(1,1)):
    return  Conv2D(filters       = conv2d_filters, 
                   kernel_size   = conv2d_kernel, 
                   padding       = conv2d_padding,
                   strides       = conv2d_stride,
                   dilation_rate = conv2d_dilation_rate,

                   kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer=regularizers.l2(1e-5),
                   bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer=regularizers.l2(1e-5)
                  )(x)

def l_act(x, act_type, name=''):
    return Activation(act_type, name=name)(x)

def l_avg_pool2d(x, pool_size=(2,2), pool_strides=(2,2), pool_padding='valid'):
    return AveragePooling2D(pool_size = pool_size, 
                            strides   = pool_strides,
                            padding   = pool_padding
                           )(x)

def l_max_pool2d(x, pool_size=(2,2), pool_strides=(2,2), pool_padding='valid'):
    return MaxPooling2D(pool_size = pool_size, 
                        strides   = pool_strides,
                        padding   = pool_padding
                       )(x)

def l_dense(x, den_unit):
    return Dense(units = den_unit,
                 activation=None,
                 use_bias=True,

                 kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                 kernel_regularizer=regularizers.l2(1e-5),
                 bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                 bias_regularizer=regularizers.l2(1e-5)
                )(x)

def l_drop(x, drop_rate):
    return Dropout(drop_rate)(x)

def l_glo_avg_pool2d(x):
    return GlobalAveragePooling2D()(x)

def l_glo_max_pool2d(x):
    return GlobalMaxPooling2D()(x)

def sig_conv2d(x,
               conv2d_filters, conv2d_kernel, conv2d_stride=(1,1), conv2d_padding='same', conv2d_dilation_rate=(1,1), 
               is_bn=True, bn_moment=0.9, bn_dim=-1,
               act_type='relu', 
               is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
               is_drop=True,drop_rate=0.1
              ):

    #batchnorm
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters, 
                 conv2d_kernel=conv2d_kernel,
                 conv2d_stride=conv2d_stride,
                 conv2d_padding=conv2d_padding,
                 conv2d_dilation_rate=conv2d_dilation_rate
                )
    #act
    x = l_act(x, act_type)
    #batchnorm
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def dob_conv2d(x,
               conv2d_filters, conv2d_kernel, conv2d_stride=(1,1), conv2d_padding='same', conv2d_dilation_rate=(1,1), 
               is_bn=True, bn_moment=0.9, bn_dim=-1,
               act_type='relu', 
               is_pool=True, pool_type='AP', pool_size=(2,2), pool_strides=(2,2), pool_padding='valid',
               is_drop=True, drop_rate=0.1
              ):

    #batchnorm 00
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 01
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters, 
                 conv2d_kernel=conv2d_kernel,
                 conv2d_stride=conv2d_stride,
                 conv2d_padding=conv2d_padding,
                 conv2d_dilation_rate=conv2d_dilation_rate
                )
    #act 01
    x = l_act(x, act_type)
    #batchnorm 01
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)
    #conv 02    
    x = l_conv2d(x, 
                 conv2d_filters=conv2d_filters, 
                 conv2d_kernel=conv2d_kernel,
                 conv2d_stride=conv2d_stride,
                 conv2d_padding=conv2d_padding,
                 conv2d_dilation_rate=conv2d_dilation_rate
                )
    #act 02
    x = l_act(x, act_type)
    #batchnorm 02
    if(is_bn):    
        x = l_batch_norm(x, bn_moment=bn_moment, bn_dim=bn_dim)

    #pooling
    if(is_pool):
        if pool_type == 'AP':
            x = l_avg_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'MP':
            x = l_max_pool2d(x, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding)
        elif pool_type == 'GAP':
            x = l_glo_avg_pool2d(x)
        elif pool_type == 'GMP':
            x = l_glo_max_pool2d(x)
        elif pool_type == 'G_MIX':
            x1 = l_glo_avg_pool2d(x)
            x2 = l_glo_max_pool2d(x)
            x  = x1 + x2
    #drop
    if is_drop:
        x = l_drop(x, drop_rate=drop_rate)

    return x

def l_inc_01(layer_in, f1, f2, f3, f4):
    # 1x1 conv - f1
    conv11 = Conv2D(f1, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 3x3 conv - f2
    conv33 = Conv2D(f2, (3,3), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)
    # 1x4 conv - f3
    conv14 = Conv2D(f3, (1,4), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(layer_in)

    # 3x3 max pooling - 1x1 conv - f4
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool_conv11 = Conv2D(f4, (1,1), padding='same', activation='relu',
                   dilation_rate      = (1,1),
                   kernel_initializer = initializers.TruncatedNormal(mean=0.0,stddev=0.1),
                   kernel_regularizer = regularizers.l2(1e-5),
                   bias_initializer   = initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                   bias_regularizer   = regularizers.l2(1e-5)
                  )(pool)

    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=-1)([conv11, conv33, conv14, pool_conv11])

    return layer_out
