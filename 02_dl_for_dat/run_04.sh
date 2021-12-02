#!/bin/bash

#SBATCH --job-name=DCASE
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2001
#SBATCH --gres=gpu:1
#source /opt/anaconda3/etc/profile.d/conda.sh

#module load use.moosefs
#module load anaconda3

#conda activate tensor2_01
#conda activate mix_01
#export HDF5_USE_FILE_LOCKING=FALSE
#export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"



#CUDA_VISIBLE_DEVICES="1,-1" python step04_trainer.py --out_dir "./11_vgg/"  --train_dir   "./../01_fea_ext/01_spec_gen_cough/11_mel/data_dev"  --test_dir    "./../01_fea_ext/01_spec_gen_cough/11_mel/data_test"  --is_training "yes"  --is_testing  "yes" --is_extract  "yes" 
python step04_trainer.py --out_dir "./11_vgg/"  --train_dir   "/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/01_fea_ext_for_dat/01_spec_gen_cough/11_MFCC/data_dev"  --test_dir    "/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/01_fea_ext_for_dat/01_spec_gen_cough/11_MFCC/data_test"  --is_training "yes"  --is_testing  "yes" --is_extract  "yes" 
