#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import re
import pulp
import pyomo
from scipy.stats import entropy
import scipy as sp
import collections



def load_p_y_obj_set(root_path, p_y_obj_set_idx):
    validation_folder = root_path + "/optimization/validation/"
    p_y_obj_set = np.load(validation_folder + "p_y_obj_set_%d.npy"%p_y_obj_set_idx)
    return p_y_obj_set


def check_results(T_norm, p_y_init, p_y_obj, num_classes, tolerance=1e-5):
    ############################
    #Validate the restrictions
    ############################
    report_failure = False
    
    #T is bounded in [0.0, 1.0]
    if np.any(T_norm<0) or np.any(T_norm>1):
        print(T_norm[T_norm<0])
        print(T_norm[T_norm>1])
        print("-- ERROR! It does not satisfy $t_{i,j}<0 ^ t_{i,j}>1, forall i,j$")
        report_failure = True

    #T is a transition matrix (row sums equals 1)
    for i in range(num_classes):
        theor  = 1.0
        observ = sum([T_norm[(i,j)] for j in range(num_classes)])
        #If for any class the restriction is not satisfied, report failure
        if np.fabs(theor-observ)>tolerance:
            print("ERROR! Not all the rows in T are probability distributions")
            report_failure = True

    #The product prod_p_init_T = P_init(Y)*T must be a probability distribution
    prod_p_init_T = sum([sum([p_y_init[i]*(T_norm[(i,j)]) for i in range(num_classes)]) for j in range(num_classes)])
    if np.fabs(1.0-prod_p_init_T)>tolerance:
        print("ERROR! The sum of P_init(Y)*(T) is not 1")
        report_failure = True
        
    #Ensure that P_init(Y)*T = P_obj(Y)
    for j in range(num_classes):
        theor = p_y_obj[j]
        observ = sum([p_y_init[i]*(T_norm[(i,j)]) for i in range(num_classes)])
        if np.fabs(theor-observ)>tolerance:
            print("ERROR! The expected a posteriori probability for one class is not correct")
            report_failure = True
    
    return report_failure



def check_V(V, num_classes, tolerance=1e-5):
    ############################
    #Validate the restrictions
    ############################
    report_failure = False
    
    if np.any(V<-tolerance) or np.any(V>(1+tolerance)):
        print("-- NO! It does not satisfy $V_{i,s,j}<0 ^ V_{i,s,j}>1, forall i,j$")
        report_failure = True
    
    #Ensure that the sum of the probs of each subset sums 1
    dim1, dim2, dim3 = V.shape
    for i_l in range(dim1):
        for i_s in range(dim2):
            labels_in_set = [l for l in range(dim3) if (i_s & (1<<l)) != 0]
            if i_l not in labels_in_set:
                if np.sum(np.abs(V[i_l,i_s,:]))>tolerance:
                    report_failure = True
                    print("Positive probabilities in %d,%d,:"%(i_l,i_s))
            else:
                prob_sum = np.sum([V[i_l,i_s,j] for j in range(num_classes) if j in labels_in_set])
                if np.fabs(prob_sum-1)>tolerance:
                    report_failure = True
                    print("Not a probability distribution in %d,%d,:"%(i_l,i_s))
                    print(V[i_l,i_s,:])
                    
    return report_failure

    

def get_row_probs_m1(T_norm, label_idx, reachable_classes):
    '''
    Returns a transition probability vector of length K, being K the possible number of classes, 
    representing the probability of transitioning from the class "label_idx" to any of the K classes.
    
    @param  T_norm:      Transition matrix
    @param  label_idx:   Class of the input sample
    @param  reachable_classes: Classes that we can reach
    @return probs:       The probability of reaching every possible class
    '''
    #Method 1: normalize the probabilities considering only those classes that can be reached
    probs_row = np.copy(T_norm[label_idx,:])
    probs_row[reachable_classes==0] = 0.0
    
    sum_reachable_probs = np.sum(probs_row[reachable_classes==1])
    if np.fabs(sum_reachable_probs)<1e-8:
        probs = np.zeros(len(probs_row))
        probs[label_idx] = 1.0 #stay in the class!
    else:
        probs = [probs_row[l]/sum_reachable_probs for l in range(len(probs_row))]
        probs = np.array(probs).flatten()
    
    return probs





def validate_T_matrix(reachability, ground_truth, prob_method, T_norm, num_classes, V_opt=None):

    # Launch experiment
    idx_vec = np.arange(reachability.shape[0])

    #Use this to simulate different validation orders:
    #np.random.seed(30)
    #np.random.shuffle(idx_vec)

    original_classes  = np.zeros(len(idx_vec),dtype=int)
    predicted_classes = np.zeros(len(idx_vec),dtype=int)
    
    for i_f in range(len(idx_vec)):
        idx = idx_vec[i_f]
        
        label_idx = np.copy(ground_truth[idx])
        original_classes[i_f] = np.copy(label_idx)

        #Reachability of the audio
        reachable_classes = np.copy(reachability[idx,:])

        #Get the probability of reaching every possible class
        if prob_method==1:
            norm_probs_row = get_row_probs_m1(T_norm, label_idx, reachable_classes)
        else:
            sys.exit("Probability transformation not supported yet!")

        #Sample one class according to the current probabilities
        pred = np.random.choice(num_classes, 1, p=norm_probs_row)
        if reachable_classes[pred]==0:
            sys.exit("Error: sampled class that is not reachable!")
        #Save the predicted class
        predicted_classes[i_f] = pred

    original_props    = np.array( [np.sum(original_classes==l)/len(original_classes)  for l in range(num_classes)] )
    adversarial_props = np.array( [np.sum(predicted_classes==l)/len(predicted_classes) for l in range(num_classes)] )
    
    fooling_rate = np.sum(original_classes!=predicted_classes)/len(original_classes)
    
    return original_props, adversarial_props, original_classes, predicted_classes





def load_and_threshold_reachability(root_path, sel_attack, sel_dataset,
                                    N_per_class, num_classes, attack_params, 
                                    max_dist_thr=None, max_iter_thr=None):
    
    cur_load_folder   = root_path + "adv_attacks/%s/experiments/" % sel_dataset
    # Load filenames and ground-truth labels
    full_filenames    = np.load(cur_load_folder + "files_N_%d.npy" % N_per_class)
    full_ground_truth = np.load(cur_load_folder + "labels_N_%d.npy" % N_per_class)

    # Load reachability of all the samples
    cur_load_folder   = root_path + "adv_attacks/%s/analysis/targeted/" % sel_dataset
    if sel_attack == "deepfool":
        cur_appendix_params = "_ov_%s.npy" % str(attack_params[0])
        norm_lp = "l2"
    elif sel_attack in ["cw_l2"]:
        cur_appendix_params = ".npy"
        norm_lp = "l2"
    elif sel_attack in ["rand_pgd", "fgsm_v2"]:
        cur_appendix_params = ".npy"
        norm_lp = "linf"
    else: sys.exit("Attack not supported!")

    # Full reachability matrix (indicates whether we reached each class or not)
    cur_load_filename = cur_load_folder + "reachability_%s_targ%s" % (sel_attack, cur_appendix_params)
    full_reachability = np.load(cur_load_filename)
    # Load the Lp norms required for each attack (l2 or linf)
    cur_load_filename = cur_load_folder + "%s_norm_vec_%s_targ%s" % (norm_lp, sel_attack, cur_appendix_params)
    full_reachability_norms = np.load(cur_load_filename)
    # Load the number of steps required for the attack
    cur_load_filename = cur_load_folder + "num_steps_vec_%s_targ%s" % (sel_attack, cur_appendix_params)
    full_reachability_steps = np.load(cur_load_filename)

    # Threshold reachability
    for i in range(full_reachability.shape[0]):
        for j in range(num_classes):
            # Check if the norm surpasses the threshold:
            if full_reachability_norms[i, j] > max_dist_thr:
                full_reachability[i, j] = 0
            # Check if the number of steps surpasses the threshold
            if full_reachability_steps[i, j] > max_iter_thr:
                full_reachability[i, j] = 0

    # Sanity check:
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    if full_reachability.shape[0] != (N_per_class*num_classes): sys.exit("Wrong rows for full_reachability")
    if full_reachability.shape[1] != num_classes:               sys.exit("Wrong cols for full_reachability")
    # Sanity check: ensure that the ground truth class is always reachable:
    for i in range(full_reachability.shape[0]):
        if full_reachability[i, full_ground_truth[i]] != 1: sys.exit("Source class not reachable!")
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    return full_filenames, full_ground_truth, full_reachability


def multistart_shuffle(i_start, N_per_class, num_classes):
    '''
    For the main evaluation (and for the sake of simplicity), the ground-truth classes are alternated,
    i.e., the true classes of the inputs are: [0,1,...k,0,1,...k,...]. I keep this format after the shuffle because
    in this way it is easier to take folds out ensuring that there is always the same number of inputs per class.
    '''
    np.random.seed(i_start)  # Set a random seed based on the multistart number
    
    reorder_classes = np.zeros((num_classes, N_per_class), dtype=int)
    for l in range(num_classes):
        cur_list = np.array( [ (i*num_classes + l) for i in range(N_per_class) ] )
        np.random.shuffle(cur_list)
        reorder_classes[l, :] = np.copy(cur_list)
        
    # Merge with the proper format (alternating per classes)
    out = np.zeros(N_per_class*num_classes, dtype=int)
    cont = 0
    for i in range(N_per_class):
        for l in range(num_classes):
            out[cont] = reorder_classes[l, i]
            # Sanity check:
            if (i*num_classes+l) != cont: sys.exit("Check the alternating strategy")
            cont += 1
    return out
