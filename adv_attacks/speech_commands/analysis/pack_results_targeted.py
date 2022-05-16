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
import numpy as np
import scipy.io.wavfile as wav
import matplotlib
import matplotlib.pyplot as plt
import random
import re


base_path = "../../../"


#Select the adversarial attack
adv_attack = "deepfool"
#adv_attack = "rand_pgd"
#adv_attack = "cw_l2"
#adv_attack = "fgsm_v2"
print("Processing adv. attack: %s"%adv_attack)



# Read labels
labels_file_path = base_path + "datasets/speech_commands_v0.02/conv_actions_labels.txt"
labels = open(labels_file_path,'r').read().split('\n')
num_classes = len(labels)

# General paths
dataset_path     = base_path + "datasets/speech_commands_v0.02/"
input_files_root = base_path + "adv_attacks/speech_commands/experiments/"


N_per_class = 1000  # Number of samples per class
N_total = N_per_class*num_classes  # Total number of samples

#Load audios to process (filenames)
input_files_path  = input_files_root + "files_N_%d.npy"%N_per_class
target_audios     = np.load(input_files_path)
#Load the corresponding ground-truth classes
input_labels_path = input_files_root + "labels_N_%d.npy"%N_per_class
ground_truth      = np.load(input_labels_path)

#Sanity check
if len(target_audios)!=N_total: sys.exit("Wrong files  size")
if len(ground_truth) !=N_total: sys.exit("Wrong labels size")


#Root path for the results corresponding to the selected experimental setup
results_root = base_path + "adv_attacks/speech_commands/%s/results_targ/"%adv_attack
print("Loading from %s"%results_root)

#Attack parameters
if adv_attack in ["deepfool"]:
    overshoot_df = 0.02  # Overshoot parameter
    max_iter = 300       # Number of iterations
elif adv_attack in ["rand_pgd"]:
    max_iter = 30  # Number of iterations
elif adv_attack in ["cw_l2"]:
    max_iter = 1000  # Number of iterations
elif adv_attack in ["fgsm_v2"]:
    max_iter = 1000  # Number of iterations
else:
    print(adv_attack)
    sys.exit("Attack not supported")

# Pats in which the results will be saved
save_path = base_path + "adv_attacks/speech_commands/analysis/targeted/"


# Variables to store the results
reachability   = np.zeros((N_total, 12), dtype=int) #1 if reached, 0 else
num_steps_vec  = np.zeros((N_total, 12), dtype=int) #number of steps to create the adv. ex.
l2_norm_vec    = np.zeros((N_total, 12))            #l2 norm of the final perturbation
secs_vec       = np.zeros((N_total, 12))            #seconds required to create the adv. ex.
linf_norm_vec     = np.zeros((N_total, 12))         #l_inf norm of the final perturbation


scale_factor = 1/(1<<15)

# Process each input
for i_f in range(N_total):
    if i_f%100==0: print(i_f)

    audio_filename = str(target_audios[i_f])
    cur_gt_label   = int(ground_truth[i_f])

    cur_results_folder = results_root + labels[cur_gt_label] + "/"

    # Retrieve file extension (reformatting the name to load the files with the results)
    filename_folder , filename_with_extension = re.match("(.+)/(.+)",audio_filename).groups()
    filename_no_extension = re.match("(.+).wav",filename_with_extension).groups()[0]
    filename_final_name = filename_folder + "_" + filename_no_extension


    # Process the results corresponding to each target class
    for target_class in range(num_classes):

        if target_class==cur_gt_label:
            #We assume that the source class is always reachable
            reachability[i_f,target_class] = 1
            #number of steps required (we assume it is zero)
            num_steps_vec[i_f,target_class] = 0
            #Perturbation norms (we assume that all of them are zero)
            l2_norm_vec[i_f,target_class] = 0.0
            linf_norm_vec[i_f,target_class] = 0.0
            #time required to create the attack (we assume it is zero)
            secs_vec[i_f,target_class] = 0.0

        else:
            # Prepare the parameters required to load the files with the results
            if adv_attack == "deepfool":
                params_at_filename = "_t_" + str(target_class) + "_ov_" + str(overshoot_df) + "_maxit_" + str(max_iter)
            elif adv_attack in ["rand_pgd", "cw_l2", "fgsm_v2"]:
                params_at_filename = "_t_" + str(target_class) + "_maxit_" + str(max_iter)
            else:
                sys.exit("Attack not supported")

            #Load the final class
            k_i      = np.load(cur_results_folder+"k_i_"+filename_final_name+params_at_filename+".npy")
            #Load the perturbation (considering  rounding constraints)
            adv_pert = np.load(cur_results_folder+"adv_pert_"+filename_final_name+params_at_filename+".npy")
            #Save the elapsed time in seconds
            secs     = np.load(cur_results_folder+"secs_"+filename_final_name+params_at_filename+".npy")
            #Load the perturbation (WITHOUT ROUNDING CONSTRAINTS)
            r_tot    = np.load(cur_results_folder+"r_tot_"+filename_final_name+params_at_filename+".npy")


            # In foolbox, if an adversarial is not found, then the output adversarial example is
            # composed of nans. Therefore r_tot will be also nans
            if np.all(np.isnan(r_tot)):
                reachability[i_f,target_class] = 0
                num_steps_vec[i_f,target_class] = 0
                l2_norm_vec[i_f,target_class] = 0.0
                linf_norm_vec[i_f,target_class] = 0.0
                secs_vec[i_f,target_class] = np.copy(secs)  # we save the time anyways...

            elif np.any(np.isnan(r_tot)):
                sys.exit("Check nans in r_tot!")  # Sanity check

            else:
                #Mark if the audio has reached the target class
                if k_i == target_class: reachability[i_f,target_class] = 1
                else:                   reachability[i_f,target_class] = 0
                #Save the results:
                l2_norm_vec[i_f,target_class]   = np.linalg.norm(adv_pert)  # Perturbation l2    norm
                linf_norm_vec[i_f,target_class] = np.max(np.abs(adv_pert))  # Perturbation l_inf norm
                secs_vec[i_f,target_class] = np.copy(secs)  # Save the time required to create the attack

                # Load and store the number of iterations needed
                if adv_attack == "deepfool":
                    loop_i   = np.load(cur_results_folder+"loop_i_"+filename_final_name+params_at_filename+".npy")
                    num_steps_vec[i_f,target_class] = np.copy(loop_i)

                elif adv_attack in ["rand_pgd", "cw_l2", "fgsm_v2"]:
                    #For these cases the number of iterations is not available, so the maximum budget is saved
                    num_steps_vec[i_f,target_class] = np.copy(max_iter)

                else:
                    sys.exit("Attack not supported")


#Save the results
if adv_attack == "deepfool":
    appendix_param_npy = "%s_targ_ov_%s.npy"%(adv_attack, str(overshoot_df))
elif adv_attack in ["rand_pgd", "cw_l2", "fgsm_v2"]:
    appendix_param_npy = "%s_targ.npy"%(adv_attack)
else:
    sys.exit("Attack not supported")

np.save(save_path + "reachability_"   + appendix_param_npy, reachability)
np.save(save_path + "num_steps_vec_"  + appendix_param_npy, num_steps_vec)
np.save(save_path + "l2_norm_vec_"    + appendix_param_npy, l2_norm_vec)
np.save(save_path + "secs_vec_"       + appendix_param_npy, secs_vec)
np.save(save_path + "linf_norm_vec_"  + appendix_param_npy, linf_norm_vec)
