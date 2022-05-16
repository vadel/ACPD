
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from os import listdir
from os.path import isfile, join
import argparse
import os.path
import sys
import struct
import time
import random


if __name__ == '__main__':


    base_path = "../../../"

    # Select the adversarial attack
    #sel_attack = "cw_l2"     #Carlini and Wagner attack
    #sel_attack = "rand_pgd"  #Projected Gradient Descent
    sel_attack = "fgsm_v2"    #Fast Gradient Sign Method


    #LOAD LABELS
    ####################
    labels_file_path = base_path + "datasets/speech_commands_v0.02/conv_actions_labels.txt"
    print("Path of the labels file: \t " + labels_file_path)
    labels = open(labels_file_path,'r').read().split('\n')


    #TARGET MODEL PATH
    ####################
    model_path = base_path + "models/my_frozen_graph_v2.pb"
    print("Path of the target model: \t " + model_path)


    #DATASET PATH
    ####################
    dataset_path = base_path + "datasets/speech_commands_v0.02/"
    print("Dataset path: %s" % dataset_path)
    

    #LOAD AUDIO FILENAMES
    ######################
    input_files_root = base_path + "adv_attacks/speech_commands/experiments/"
    print("Input files root: %s" % input_files_root)

    #Number of iterations for the attacks (see Foolbox_Attacks.py)
    if sel_attack in ["rand_pgd"]:
        maxiter = 30
        miniter = 0
    elif sel_attack in ["cw_l2"]:
        maxiter = 1000
        miniter = 0
    elif sel_attack in ["fgsm_v2"]:
        maxiter = 1000
        miniter = 0
    else:
        sys.exit("Attack not supported")

    print("%d , %d" % (maxiter, miniter))

    #Number of samples for each class
    N_per_class = 1000 

    #Root in which the results are saved
    results_root = base_path + "adv_attacks/speech_commands/%s/results_targ/" % sel_attack

    # Parameters for parallel computation (leave as is for a single execution)
    N_samples_per_launch = 1000  #Number of samples per launch (for each pair of source x target classes)
    N_launches = N_per_class//N_samples_per_launch  #Total number of launches (for each pair of source x target classes)

    #Sanity check
    if (N_launches*N_samples_per_launch) != N_per_class:
        sys.exit("Wrong launches - set a divisible batch!")

    #Process each pair of source and target classes
    for label_idx in range(12):

        print("Source class: %d" % label_idx)

        for target_class in range(12):

            if label_idx == target_class:
                continue

            minidx = 0
            maxidx = N_samples_per_launch

            for i_launch in range(N_launches):
                results_folder = results_root + labels[label_idx] + "/"
                print("Results save folder: " + results_folder)
                input_files_folder = input_files_root + labels[label_idx] + "/"
                print("Input files folder: " + input_files_folder)

                cmd = "python3 Foolbox_Attacks.py "
                cmd = cmd + " -i "  + input_files_folder + "files_%s_N_%d.npy" % (labels[label_idx], N_per_class)
                cmd = cmd + " -l "  + str(label_idx)
                cmd = cmd + " -m "  + model_path
                cmd = cmd + " -d "  + dataset_path
                cmd = cmd + " -lf " + labels_file_path
                cmd = cmd + " -s "  + results_folder
                cmd = cmd + " -minidx "       + str(minidx)
                cmd = cmd + " -maxidx "       + str(maxidx)
                cmd = cmd + " -maxiter "      + str(maxiter)
                cmd = cmd + " -miniter "      + str(miniter)
                cmd = cmd + " -sel_attack "   + sel_attack
                cmd = cmd + " -target_class " + str(target_class)

                #Launch process
                print(cmd)
                os.system(cmd)

                print("Launched %d-%d" % (minidx, maxidx))

                minidx = minidx + N_samples_per_launch
                maxidx = maxidx + N_samples_per_launch

    print("All processes have been successfully launched")



