import numpy as np
import os

class hypara(object):

    def __init__(self):

        #---- Original Dataset Directory
        self.audio_data_dir = '/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/for_dat/Second_DiCOVA_Challenge_Dev_Data_Release/AUDIO/cough'
        self.dev_csv        = '/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/for_dat/Second_DiCOVA_Challenge_Dev_Data_Release/dev_metedata.csv'
        self.train_csv      = '/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/for_dat/Second_DiCOVA_Challenge_Dev_Data_Release/LISTS/train_0.csv'
        self.eva_csv        = '/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/for_dat/Second_DiCOVA_Challenge_Dev_Data_Release/LISTS/val_0.csv'

        self.test_audio_data_dir = '/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/for_dat/Second_DiCOVA_Challenge_Test_Data_Release/AUDIO/cough'
        self.test_csv            = '/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/for_dat/Second_DiCOVA_Challenge_Test_Data_Release/test_metedata.csv'
      
        #---- Create Label dic
        self.label_dict     = dict(p=0, n=1)

        #---- Para for generating spectrogram
        self.res_fs = 16000

        #01/ log-mel
        self.mel_n_bin = 64
        self.mel_n_win = 2048 
        self.mel_n_fft = 4096
        self.mel_f_min = 10
        self.mel_f_max = None
        self.mel_htk   = False
        self.mel_n_hop = 650

        #02/ gam
        self.gam_n_bin = 128
        self.gam_n_win = 2048 
        self.gam_n_fft = 4096
        self.gam_f_min = 10
        self.gam_f_max = None
        self.gam_htk   = False
        self.gam_n_hop = 650
        
        #02/ cqt
        self.cqt_bins_per_octave = 24
        self.cqt_n_bin = 128
        self.cqt_f_min = 10
        self.cqt_n_hop = 672
        
        #03/ other para
        self.eps  = np.spacing(1)
        self.nT   = 128    #Time resolution
        self.nF   = 128    #Frequency resolution

        #---- Para for generator and training
        self.batch_size    = 10    #The number of recording, each which have 4 batches of 128x128
        self.start_batch   = 300     #batch from 0 to 200: for evaluating; batch from 201 to the end: for training
        self.learning_rate = 1e-3
        self.is_mixup      = True
        self.check_every   = 20
        self.class_num     = 10
        self.epoch_num     = 300
