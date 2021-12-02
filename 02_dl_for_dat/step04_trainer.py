#---- general packages
import numpy as np
import os
import argparse
import math
import scipy.io
import re
import time
import datetime
import sys
import tensorflow as tf
from sklearn import datasets, svm, metrics

#----- generator
sys.path.append('./02_models/')
#from baseline import *
from inception01 import *
from vgg import *
from util import *
from generator import *
from hypara import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#----- main
def main():
    print("\n ==================================================================== SETUP PARAMETERS...")
    print("-------- PARSER:")
    parser = init_argparse()
    args   = parser.parse_args()

    OUT_DIR          = args.out_dir
    IS_TRAINING      = args.is_training
    IS_TESTING       = args.is_testing
    IS_EXTRACT    = args.is_extract

    print("-------- Hyper parameters:")
    NF_AUD           = hypara().nF_aud  
    NT_AUD           = hypara().nT_aud
    NC_AUD           = hypara().nC_aud

    BATCH_SIZE       = hypara().batch_size
    START_BATCH      = hypara().start_batch
    LEARNING_RATE    = hypara().learning_rate
    IS_MIXUP           = hypara().is_mixup
    CHECKPOINT_EVERY = hypara().check_every
    N_CLASS          = hypara().class_num
    NUM_EPOCHS       = hypara().epoch_num
    
    #Setting directory
    print("\n =============== Directory Setting...")
    stored_dir = os.path.abspath(os.path.join(os.path.curdir, OUT_DIR))
    print("+ Writing to {}\n".format(stored_dir))

    best_model_dir = os.path.join(stored_dir, "model")
    print("+ Best model Dir: {}\n".format(best_model_dir))
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_file = os.path.join(best_model_dir, "best_model.h5")

    #Random seed
    tf.random.set_seed(0) #V2
    
    #Instance model or reload an available model
    if os.path.isfile(best_model_file):
        model = tf.keras.models.load_model(best_model_file)
        with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
            text_file.write("Latest model is loaded from: {} ...\n".format(best_model_dir))
    else:
        #model=inception01(nF_aud=NF_AUD, nT_aud=NT_AUD, nC_aud=NC_AUD, nClass=N_CLASS, chanDim=-1)
        model=vgg(nF_aud=NF_AUD, nT_aud=NT_AUD, nC_aud=NC_AUD, nClass=N_CLASS, chanDim=-1)
        with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
            text_file.write("New model instance is created...\n")
    model.summary()
    #exit()

    ## initialize the optimizer and compile the model
    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    #losses = {"output": "categorical_crossentropy"}

    #model.compile(loss=tf.keras.losses.kullback_leibler_divergence, optimizer=opt, metrics=["accuracy"])   
    #model.compile(loss=[cus_loss, cus_loss, cus_loss], optimizer=opt, metrics=["accuracy"])   
    model.compile(loss=tf.keras.losses.kullback_leibler_divergence, optimizer=opt, metrics=["accuracy"])   

    old_auc = 0

    #test_threshold   = 0.7
    test_threshold   = 0.1
    generator_ins    = generator(args.train_dir, args.test_dir)
    n_valid          = generator_ins.get_file_num(generator_ins.test_dir)
    batch_num        = generator_ins.get_batch_num(BATCH_SIZE)
    #print(batch_num) = (793/40) + 1 = 20
    #exit()

    if IS_TRAINING == 'yes':
        for nEpoch in range(NUM_EPOCHS):
            print("\n=======================  Epoch is", nEpoch, ";============================")
            #for nBatchTrain in range(START_BATCH, batch_num):
            START_BATCH = 0
            batch_num = 3
            
            for nBatchTrain in range(START_BATCH, batch_num):
                x_train_batch, y_train_batch, n_image = generator_ins.get_batch(nBatchTrain, BATCH_SIZE, IS_MIXUP) #    
                #print(np.shape(x_train_batch))
                #exit()
                [train_loss, train_acc] = model.train_on_batch(x_train_batch, y_train_batch,reset_metrics=True)
                if (nBatchTrain % CHECKPOINT_EVERY == 0):  
                    print("Epoch: {}, TRAIN Accuracy:{}".format(nEpoch,train_acc))
                    with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                        text_file.write("Epoch:{}, TRAIN ACC:{} \n".format(nEpoch, train_acc))

                    if(train_acc >= test_threshold):   
                            file_valid_acc   = 0
                            fuse_matrix      = np.zeros([N_CLASS, N_CLASS])
                            valid_metric_reg = np.zeros(n_valid)
                            valid_metric_exp = np.zeros(n_valid)
     
                            #for nFileValid in range(0,n_valid):
                            for nFileValid in range(0,10):
                                x_valid_batch, valid_file_name = generator_ins.get_single_file(nFileValid, args.test_dir)
                                #print(np.shape(x_valid_batch))
                                #exit()
                                #expected  
                                class_name = valid_file_name.split('.')[0].split('_')[-1] #full
                                valid_res_exp  = hypara().label_dict[class_name]
                                #recognized  
                                #print(type(x_valid_batch))
                                #x_valid_batch = x_valid_batch[0:2,:,:,:].astype(np.float16)   #0:2 --> only check 0:2
                                x_valid_batch = x_valid_batch.astype(np.float16)   
                                valid_end_output = model.predict(x_valid_batch)
    
                                #print(np.shape(valid_end_output))
                                #exit()

                                sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
                                valid_res_reg        = np.argmax(sum_valid_end_output)
                                
                                #Compute acc
                                valid_metric_reg[nFileValid] = int(valid_res_reg)
                                valid_metric_exp[nFileValid] = int(valid_res_exp)
     
                                #For general report
                                fuse_matrix[valid_res_exp, valid_res_reg] = fuse_matrix[valid_res_exp, valid_res_reg] + 1
                                if(valid_res_reg == valid_res_exp):
                                    file_valid_acc = file_valid_acc + 1
 
                            # For general report
                            file_valid_acc  = file_valid_acc*100/n_valid
                            #print("Testing Accuracy: {} % \n".format(file_valid_acc))   
     
                            #for sklearn metric
                            #print("Classification report for classifier \n%s\n"
                            rp = metrics.classification_report(valid_metric_exp, valid_metric_reg)
                            cm = metrics.confusion_matrix(valid_metric_exp, valid_metric_reg)

                            tn, fp, fn, tp = metrics.confusion_matrix(valid_metric_exp, valid_metric_reg).ravel()
                            specificity    = tn/(tn+fp)
                            sensitivity    = tp/(tp+fn)

                            fpr, tpr, thresholds = metrics.roc_curve(valid_metric_exp, valid_metric_reg)
                            auc = metrics.auc(fpr, tpr)

                            #print("Confusion matrix:\n%s" % cm)
     

                            
                            if auc > old_auc:
                                old_auc = auc 
                                model.save(best_model_file)
                                with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                                    text_file.write("Save best model at Epoch: {}; AUC Score:{} \n".format(nEpoch, auc))

                                with open(os.path.join(stored_dir,"valid_acc_log.txt"), "a") as text_file:
                                    text_file.write("========================== VALIDATING ONLY =========================================== \n\n")
                                    text_file.write("On File Final Accuracy:  {}%\n".format(file_valid_acc))
                                    text_file.write("{0} \n".format(rp))
                                    text_file.write("{0} \n".format(cm))
                                    text_file.write("SPEC.: {0} \n".format(specificity))
                                    text_file.write("SEN.:  {0} \n".format(sensitivity))
                                    text_file.write("AUC:   {0} \n".format(auc))
                                    text_file.write("========================================================================== \n\n")

    #--------- Extract embbeding
    if (IS_EXTRACT == 'yes'):
        final_layer_stored_matrix = np.zeros([n_valid, N_CLASS]) #file_num x nClass
    
        final_layer_res_dir = os.path.abspath(os.path.join(stored_dir, "01_pre_res"))
        if not os.path.exists(final_layer_res_dir):
            os.makedirs(final_layer_res_dir)
        final_layer_file = os.path.abspath(os.path.join(final_layer_res_dir, "pre_train_res"))
        file_name_list = []       
        for nFileValid in range(0,n_valid):
            x_valid_batch, valid_file_name = generator_ins.get_single_file(nFileValid, args.test_dir)
    
            file_name_list.append(valid_file_name)
            x_selected = x_valid_batch.astype(np.float16)
            valid_end_output = model.predict(x_selected)
            sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
            final_layer_stored_matrix[nFileValid,:] = sum_valid_end_output
        np.savez(final_layer_file, matrix=final_layer_stored_matrix, file_name=file_name_list)


def update_lr(model, epoch):
    if epoch >= 100:
        new_lr = 0.0001
        model.optimizer.lr.assign(new_lr)

def cus_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))


def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --out_dir XXX --dev_dir XXX --train_dir XXX --eva_dir XXX --test_dir XXX",
        description="Set directory of spectrogram of dev/train/eva/test sets" 
    )
    parser.add_argument(
        "--out_dir", required=True,
        help=' --outdir <output directory>'
    )
    parser.add_argument(
        "--train_dir", required=True,
        help=' --train_dir <train spectrogram directory>'
    )
    parser.add_argument(
        "--test_dir", required=True,
        help=' --test_dir <test spectrogram directory>'
    )
    parser.add_argument(
        "--is_training", required=True,
        help=' --is_training <yes/no>'
    )
    parser.add_argument(
        "--is_testing", required=True,
        help=' --is_testing <yes/no>'
    )
    parser.add_argument(
        "--is_extract", required=True,
        help=' --is_extract <yes/no>'
    )
    return parser


if __name__ == "__main__":
    main()
