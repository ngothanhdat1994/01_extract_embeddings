import numpy as np
import os
from itertools import islice
import sys
import re
from natsort import natsorted, ns
from hypara import *
import random
import scipy

class generator(object):

    def __init__(self, train_dir, test_dir):  

        self.train_dir      = train_dir
        self.test_dir       = test_dir

        self.class_num      = hypara().class_num
        self.label_dict     = hypara().label_dict


    def get_single_file (self, file_num, data_dir): 
        file_list = self.get_file_list(data_dir)
        file_name     = file_list[file_num]
        file_open     = os.path.join(data_dir, file_name)
        o_data        = np.load(file_open)  
        #file_str      = scipy.io.loadmat(file_open)  
        #o_data        = file_str['final_data']
        #nF, nT        = np.shape(o_data)
        #o_data        = np.reshape(o_data, (1,nF,nT,1))
        return o_data, file_name

    def get_batch (self, batch_num, batch_size, is_mixup):

        org_class_list = os.listdir(self.train_dir)
        class_list = []  #remove .file
        for nClass in range(0,len(org_class_list)):
            isHidden=re.match("\.",org_class_list[nClass])
            if (isHidden is None):
                class_list.append(org_class_list[nClass])
        class_num  = len(class_list)
        class_list = sorted(class_list)

        nImage = 0
        for class_mem in class_list:
            file_dir = os.path.join(self.train_dir, class_mem)

            org_file_list = os.listdir(file_dir)
            file_list = []  #remove .file
            for nFile in range(0,len(org_file_list)):
                isHidden=re.match("\.",org_file_list[nFile])
                if (isHidden is None):
                    file_list.append(org_file_list[nFile])
            file_num  = len(file_list)
            file_list = sorted(file_list)
            train_file_id  = np.random.RandomState(seed=42).permutation(file_num)


            for ind in range(batch_num*batch_size, (batch_num+1)*batch_size):
                if ind >= file_num:
                    mul = int(ind/file_num)
                    ind = ind - mul*file_num
                # open file
                file_name = file_list[train_file_id[ind]]
                file_open = os.path.join(self.train_dir, class_mem, file_name)
   
                #create label
                class_name = file_name.split('.')[0].split('_')[-1]  #patch
                nClass = self.label_dict[class_name]                                        
                expectedClass = np.zeros([1,self.class_num])
                expectedClass[0,nClass] = 1

                ##create data with vector input
                #one_vector = np.reshape(np.load(file_open),(1,-1))  
                #if (nImage == 0):
                #   seq_x = one_vector
                #   seq_y = expectedClass
                #else:            
                #   seq_x = np.concatenate((seq_x, one_vector), axis=0)  
                #   seq_y = np.concatenate((seq_y, expectedClass), axis=0)  

                #create data with image input
                one_image = np.load(file_open)  
                #file_str = scipy.io.loadmat(file_open)  
                #one_image = file_str['final_data']
                #nF, nT    = np.shape(one_image)
                #one_image = np.reshape(one_image, (1,nF,nT,1))
                if (nImage == 0):
                   seq_x = one_image
                   seq_y = expectedClass
                else:            
                   seq_x = np.concatenate((seq_x, one_image), axis=0)  
                   seq_y = np.concatenate((seq_y, expectedClass), axis=0)  

                nImage += 1
        #print(np.shape(seq_x))
        #print(np.shape(seq_y))
        #exit()

        if is_mixup:
            o_data, o_label = self.data_aug(seq_x, seq_y, 0.4)
        else:
            o_data  = seq_x
            o_label = seq_y

        return o_data, o_label, nImage

    def get_batch_num(self, batch_size):

        org_class_list = os.listdir(self.train_dir)
        class_list = []  #remove .file
        for nClass in range(0,len(org_class_list)):
            isHidden=re.match("\.",org_class_list[nClass])
            if (isHidden is None):
                class_list.append(org_class_list[nClass])
        class_num  = len(class_list)
        class_list = sorted(class_list)

        max_file_num = 0
        for class_mem in class_list:
            file_dir = os.path.join(self.train_dir, class_mem)

            org_file_list = os.listdir(file_dir)
            file_list = []  #remove .file
            for nFile in range(0,len(org_file_list)):
                isHidden=re.match("\.",org_file_list[nFile])
                if (isHidden is None):
                    file_list.append(org_file_list[nFile])
            file_num  = len(file_list)
            if file_num > max_file_num:
                max_file_num = file_num
                
        return int(max_file_num/batch_size) + 1
    
    def get_file_num(self, data_dir):
        return len(os.listdir(data_dir))

    def get_file_list(self, data_dir):
        file_list = []
        org_file_list = os.listdir(data_dir)
        for i in range(0,len(org_file_list)):
           isHidden=re.match("\.",org_file_list[i])
           if (isHidden is None):
              file_list.append(org_file_list[i])
        #natsorted(file_list)
        file_list.sort()
        return file_list

    #---- online mixup data augmentation
    def data_aug(self, i_data, i_label, beta=0.4):
        half_batch_size = round(np.shape(i_data)[0]/2)
        #print(half_batch_size, np.shape(i_data))

        #x1  = i_data[:half_batch_size,:] #for input vector
        #x2  = i_data[half_batch_size:,:]
        x1  = i_data[:half_batch_size, :, :, :] #for input image
        x2  = i_data[half_batch_size:, :, :, :]

        # frequency/time masking is only for image
        #for j in range(x1.shape[0]):
        #   # spectrum augment
        #   for c in range(x1.shape[3]):
        #       x1[j, :, :, c] = self.frequency_masking(x1[j, :, :, c])
        #       x1[j, :, :, c] = self.time_masking(x1[j, :, :, c])
        #       x2[j, :, :, c] = self.frequency_masking(x2[j, :, :, c])
        #       x2[j, :, :, c] = self.time_masking(x2[j, :, :, c])

        y1  = i_label[:half_batch_size,:]
        y2  = i_label[half_batch_size:,:]

        # Beta dis
        b   = np.random.beta(beta, beta, half_batch_size)
        #X_b = b.reshape(half_batch_size, 1) #for vector input
        X_b = b.reshape(half_batch_size, 1, 1, 1) #for image input
        y_b = b.reshape(half_batch_size, 1)

        xb_mix   = x1*X_b     + x2*(1-X_b)
        xb_mix_2 = x1*(1-X_b) + x2*X_b

        yb_mix   = y1*y_b     + y2*(1-y_b)
        yb_mix_2 = y1*(1-y_b) + y2*y_b

        # Uniform dis
        l   = np.random.random(half_batch_size)
        #X_l = l.reshape(half_batch_size, 1) #for vector input
        X_l = l.reshape(half_batch_size, 1, 1, 1) #for image input
        y_l = l.reshape(half_batch_size, 1)

        xl_mix   = x1*X_l     + x2*(1-X_l)
        xl_mix_2 = x1*(1-X_l) + x2*X_l

        yl_mix   = y1* y_l    + y2 * (1-y_l)
        yl_mix_2 = y1*(1-y_l) + y2*y_l

        #o_data     = np.concatenate((xb_mix,    x1,    xl_mix,    xb_mix_2,    x2,    xl_mix_2),    0)
        #o_label    = np.concatenate((yb_mix,    y1,    yl_mix,    yb_mix_2,    y2,    yl_mix_2),    0)
        o_data     = np.concatenate((xb_mix,    x1,    xb_mix_2,    x2),    0)
        o_label    = np.concatenate((yb_mix,    y1,    yb_mix_2,    y2),    0)
        #o_data     = np.concatenate((xb_mix,    xl_mix,    xb_mix_2,    xl_mix_2),    0)
        #o_label    = np.concatenate((yb_mix,    yl_mix,    yb_mix_2,    yl_mix_2),    0)

        return o_data, o_label

    def frequency_masking(self, mel_spectrogram, frequency_masking_para=10, frequency_mask_num=1):
        fbank_size = mel_spectrogram.shape
    
        for i in range(frequency_mask_num):
            f = random.randrange(0, frequency_masking_para)
            f0 = random.randrange(0, fbank_size[0] - f)
    
            if (f0 == f0 + f):
                continue
    
            mel_spectrogram[f0:(f0+f),:] = 0
        return mel_spectrogram
       
       
    def time_masking(self, mel_spectrogram, time_masking_para=10, time_mask_num=1):
        fbank_size = mel_spectrogram.shape
    
        for i in range(time_mask_num):
            t = random.randrange(0, time_masking_para)
            t0 = random.randrange(0, fbank_size[1] - t)
    
            if (t0 == t0 + t):
                continue
    
            mel_spectrogram[:, t0:(t0+t)] = 0
        return mel_spectrogram        
