#!/usr/bin/env python3
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

    root_path = "../"

    # Select adversarial attack
    sel_attack = "deepfool"   #DeepFool
    #sel_attack = "cw_l2"     #Carlini & Wagner
    #sel_attack = "fgsm_v2"   #Fast Gradient Sign Method
    #sel_attack = "rand_pgd"  #Projected Gradient Descent

    # Select method
    sel_method  = "m1"    # ** Method 1 - Agnostic Method (AM)
    #sel_method = "m2"    # ** Method 2 - Upper-bound Method (UBM)
    #sel_method = "m3"    # ** Method 3 - Element-wise Transformation Method (EWTM)
    #sel_method = "m4"    # ** Method 4 - Chain-rule Method (CRM)
    #sel_method = "mab"   # ** Maximum Approximation Baseline (MAB)
    #sel_method = "mfrb"  # ** Maximum Fooling Rate Baseline (MFRB)
    # Variations:
    #sel_method = "m4_1"  # Method 4 (CRM) w/o Laplace  &  w/o fixing P({y_i}|y_i) to zero.
    #sel_method = "m4_2"  # Method 4 (CRM) w/  Laplace but w/o fixing P({y_i}|y_i) to zero.

    # Select the set of target distributions
    p_y_obj_set_idx  = 1  # 100 random Dirichlet distributions
    #p_y_obj_set_idx = 2  # Uniform distribution

    # Select dataset
    sel_dataset = "speech_commands"

    if sel_dataset == "speech_commands":
        # Load problem labels
        labels_file_path = root_path + "datasets/speech_commands_v0.02/conv_actions_labels.txt"
        print("Path of the labels file: \t " + labels_file_path)
        labels = open(labels_file_path, 'r').read().split('\n')
    else:
        sys.exit("Supported datasets: [speech_commands]")


    #Evaluation parameters
    N_per_class = 1000
    N_per_class_train = 500  #By default, a 2-fold cv will be executed
    min_multistart = 0
    max_multistart = 50
    N_multistarts = max_multistart - min_multistart #Number of k-fold cv "multistarts"
    # For parallel launching purposes (leave as is for a single execution):
    N_multistarts_per_launch = 50  #How many starts will be executed in each process (must be ==N_multistarts for a single execution)
    N_launches_multistart = int(N_multistarts//N_multistarts_per_launch)
    if N_multistarts % N_multistarts_per_launch != 0: sys.exit("N_multistarts_per_launch must be a divisor of N_multistarts")

    # RESULTS ROOT
    results_root = root_path + "optimization/%s/%s/results_%s/" % (sel_dataset, sel_attack, sel_method)
    save_path = results_root + "p_y_obj_set_%d/" % p_y_obj_set_idx
    save_path = save_path    + "N_train_%d/" % N_per_class_train


    # Maximum number of iterations
    if sel_attack == "deepfool":
        max_iter_thr = 30
    elif sel_attack in ["rand_pgd"]:
        max_iter_thr = 30
    elif sel_attack in ["cw_l2"]:
        max_iter_thr = 1000
    elif sel_attack in ["fgsm_v2"]:
        max_iter_thr = 1000
    else:
        sys.exit("Attack not supported")
    print("Max_iter_thr : %d" % max_iter_thr)
    # Parameter for DeepFool only
    overshoot_df = 0.02

    get_row_probs_method = 1

    #Launch the attack evaluation for the following distortion thresholds:
    for max_dist_thr in [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.15]:

        cur_min_multistart = min_multistart
        cur_max_multistart = cur_min_multistart + N_multistarts_per_launch

        for i_start in range(N_launches_multistart):

            print("minstart-maxstart: %d-%d" % (cur_min_multistart, cur_max_multistart))

            cmd = "python3 ACPD_experiments_complete.py"
            cmd = cmd + " -dataset "  + sel_dataset
            cmd = cmd + " -attack "   + sel_attack
            cmd = cmd + " -method "   + sel_method
            cmd = cmd + " -r "        + root_path
            cmd = cmd + " -lf "       + labels_file_path
            cmd = cmd + " -s "        + save_path
            cmd = cmd + " -pyidx "    + str(p_y_obj_set_idx)
            cmd = cmd + " -N_per_class "       + str(N_per_class)
            cmd = cmd + " -N_per_class_train " + str(N_per_class_train)
            cmd = cmd + " -min_multistart "    + str(cur_min_multistart)
            cmd = cmd + " -max_multistart "    + str(cur_max_multistart)
            cmd = cmd + " -maxdist_thr "       + str(max_dist_thr)
            cmd = cmd + " -maxiter_thr "       + str(max_iter_thr)
            cmd = cmd + " -probs_method "      + str(get_row_probs_method)
            cmd = cmd + " -df_ov "             + str(overshoot_df)

            #Launch process
            print(cmd)
            os.system(cmd)

            cur_min_multistart = cur_min_multistart + N_multistarts_per_launch
            cur_max_multistart = cur_min_multistart + N_multistarts_per_launch


    print("All processes have been successfully launched")
