import numpy as np
import os

class hypara(object):

    def __init__(self):

        #---- Create Label dic
        self.label_dict     = dict(p=1, n=0)


        #---- Para for generator and training
        self.eps   = np.spacing(1)   # 2.220446049250313e-16
        self.nF_aud    = 64    #Frequency resolution From FE step
        self.nT_aud    = 313    #Time resolution From FE step
        self.nC_aud    = 1      #Frequency resolution

        self.batch_size    = 40    #The number of recording, each which have 4 batches of 128x128
        self.start_batch   = 0     #batch from 0 to 200: for evaluating; batch from 201 to the end: for training
        self.learning_rate = 1e-4  #-3
        self.is_mixup      = False
        self.check_every   = 2
        self.class_num     = len(self.label_dict)
        #self.epoch_num     = 300
        self.epoch_num     = 3
