#sys.path.insert(0, '/home/cug/ldp7/.local/lib/python2.7/site-packages/librosa/')
import os
import sys
import re
import numpy as np
import librosa
import soundfile as sf
from itertools import islice
import argparse

#----- import hyper-parameters for extracting spectrogram
#sys.path.insert(0, './../../')
from hypara import *

#------------------------------------------------------------ FUNCTIONS DEFINE -----------------------------
def get_data_mel(dataset, outdir, is_del, dur):

   #---- Create directory for storing
   if dataset == 'test':
       audio_data_dir = hypara().test_audio_data_dir
   else:
       audio_data_dir = hypara().audio_data_dir

   type_aud = audio_data_dir.split('/')[-1] #cough/breath/speech

   store_dir = "./01_spec_gen_" + type_aud
   if not os.path.exists(store_dir):
      os.makedirs(store_dir)

   store_dir = os.path.join(store_dir, outdir)
   if not os.path.exists(store_dir):
      os.makedirs(store_dir)
   
   if dataset == 'dev':
      print("-------- Dev set:")
      store_dir = os.path.join(store_dir, 'data_dev')
      meta_csv = hypara().dev_csv
      label_csv = hypara().dev_csv
   elif dataset == 'train':
      print("-------- Train subset:")
      store_dir = os.path.join(store_dir, 'data_train')
      meta_csv = hypara().train_csv
      label_csv = hypara().dev_csv
   elif dataset == 'eva':
      print("-------- Eva subset:")
      store_dir = os.path.join(store_dir, 'data_eva')
      meta_csv = hypara().eva_csv
      label_csv = hypara().dev_csv
   elif dataset == 'test':
      print("-------- Test subset:")
      store_dir = os.path.join(store_dir, 'data_test')
      meta_csv = hypara().test_csv
      label_csv = hypara().test_csv
   else:
       print('\n\n -----------------ERROR: NO SUBSET DATA IS SELLECTED\n')
       exit()

   if not os.path.exists(store_dir):
      os.makedirs(store_dir)

   #---- Get hyper-parameters and data directory
   res_fs= hypara().res_fs 
   n_bin = hypara().mel_n_bin
   n_win = hypara().mel_n_win 
   n_hop = hypara().mel_n_hop 
   n_fft = hypara().mel_n_fft 
   f_min = hypara().mel_f_min 
   f_max = hypara().mel_f_max 
   htk   = hypara().mel_htk   
   eps   = hypara().eps   
   nT    = hypara().nT
   nF    = hypara().nF

   #---- Get file list
   file_name_list = []  
   if dataset == 'dev' or dataset == 'test':  #all dev set
       org_file_name_list = os.listdir(audio_data_dir)
       for i in range(0,len(org_file_name_list)):
          isHidden=re.match("\.",org_file_name_list[i])
          if (isHidden is None):
             file_name_list.append(org_file_name_list[i].split('.')[0])
   else: #train/eval/test set
        with open(meta_csv, 'r') as csv_r:
            for line in islice(csv_r, 1, None):
                file_name = line.split('\n')[0]
                file_name_list.append(file_name)

   #---- Get label
   label_dict = {}
   for file_name in file_name_list:
        with open(label_csv, 'r') as csv_r:
            for line in islice(csv_r, 1, None):
                line = line.split('\n')[0]
                if re.search(file_name, line):
                    if re.search(',n,', line):
                        label_dict[file_name] = '_n'
                    elif re.search(',p,', line):
                        label_dict[file_name] = '_p'
                    else:
                        print('---- ERROR: CANNOT FIND LABEL\n')
                        exit()
   if len(label_dict) != len(file_name_list):
       print('------------ERROR: MISMATCH NUMBER OF LABEL & FILE')
       exit()

   #---- Generate mel/log-mel spectrogram for each file in file_name_list
   for nFile in range(0, len(file_name_list)):
       file_name  = file_name_list[nFile]
       file_open  = audio_data_dir + '/' + file_name + '.flac'

       org_wav, org_fs = sf.read(file_open)
       org_dur = len(org_wav)/org_fs
       wav = org_wav
       # make sure larger than 10 seconds
       if org_dur < dur:
           while True:
               wav = np.concatenate((wav, org_wav),0)
               if len(wav) > dur*org_fs:
                   break

       ### new_dur = len(wav)/org_fs
       ### if new_dur%dur >= 5:
       ###     split_num = np.floor(new_dur/dur) + 1    
       ### else:
       ###     split_num = np.floor(new_dur/dur)     

       ### print(org_dur, new_dur, split_num)
       split_num=1
       for i in range(0, int(split_num)):
           ###if i == int(split_num)-1:
           ###    ed_pt = len(wav)
           ###    st_pt = len(wav) - dur*org_fs
           ###else:
           st_pt = i*dur*org_fs
           ed_pt = (i+1)*dur*org_fs

           wav_split = wav[int(st_pt): int(ed_pt)]

           if dataset == 'test':
               des_dir  = store_dir
           else:
               if label_dict[file_name] == '_n':
                   des_dir  = os.path.join(store_dir, 'neg')
               else:
                   des_dir  = os.path.join(store_dir, 'pos')

           if not os.path.exists(des_dir):
              os.makedirs(des_dir)

           file_des = os.path.join(des_dir, file_name+label_dict[file_name]+'.wav')
           librosa.output.write_wav(file_des, wav_split, org_fs)
           #print(np.shape(wav_split))
           #exit()
 
   if dataset == 'dev':
       print ("==================== Done extracting for all development \n\n")   
   elif dataset == 'train':
       print ("==================== Done extracting for training set \n\n")   
   elif dataset == 'eva':
       print ("==================== Done extracting for evaluating set \n\n")   
   elif dataset == 'test':
       print ("==================== Done extracting for testing set \n\n")   

def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --dataset 'dev' --outdir '11_mel_cor' --cor 'yes' --delta 'yes'",
        description="--> --dataset: To generate log-mel spectrogram: dev/train/eva/test;" 
                    "Use terms of 'train'/'eva'/'test' for the other subsets;"
                    "Use terms of 'all' for all subsets"
                    "--> --cor: Correct spectrogram or not: yes/no"
                    "--> --delta: Apply delta on spectrogram or not: yes/no"
    )
    parser.add_argument(
        "--dataset", required=True,
        help='Choose dev, train, eva, or test set'
    )
    parser.add_argument(
        "--outdir", required=True,
        help='Choose out directory'
    )
    parser.add_argument(
        "--delta", required=True,
        help='Apply delta on spectrogram: yes or no'
    )
    parser.add_argument(
        "--dur", required=True,
        help='Apply delta on spectrogram: yes or no'
    )
    return parser

def main():
    print("-------- PARSER:")
    parser = init_argparse()
    args   = parser.parse_args()
    if args.dataset == 'all': 
        print("-------- STARTING TO GENERATE LOG-MEL SPECTROGRAM FOR ALL SUBSET:")
        get_data_mel('dev',   args.outdir, args.delta, args.dur)
        get_data_mel('train', args.outdir, args.delta, args.dur)
        get_data_mel('eva',   args.outdir, args.delta, args.dur)
        get_data_mel('test',  args.outdir, args.delta, args.dur)
    elif args.dataset == 'dev' or args.dataset == 'train' or args.dataset == 'eva' or args.dataset == 'test':
        print("-------- STARTING TO GENERATE LOG-MEL SPECTROGRAM:")
        get_data_mel(args.dataset, args.outdir, args.delta, int(args.dur))
    else:
        parser.print_help()

#------------------------------------------------------------ MAIN FUNCTION -----------------------------
if __name__ == "__main__":
    main()

