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

from acpd_utils import load_p_y_obj_set
from acpd_utils import check_results
from acpd_utils import check_V
from acpd_utils import get_row_probs_m1
from acpd_utils import validate_T_matrix
from acpd_utils import load_and_threshold_reachability
from acpd_utils import multistart_shuffle

from acpd_methods import generate_T_MAB       # ** Maximum Approximation Baseline (MAB)
from acpd_methods import generate_T_MFRB      # ** Maximum Fooling Rate Baseline (MFRB)
from acpd_methods import generate_T_agnostic  # ** Method 1 - Agnostic Method (AM)
from acpd_methods import generate_T_method_2  # ** Method 2 - Upper-bound Method (UBM)
from acpd_methods import generate_T_method_3  # ** Method 3 - Element-wise Transformation Method (EWTM)
from acpd_methods import compute_subset_probs
from acpd_methods import generate_T_method_4  # ** Method 4 - Chain-rule Method (CRM)


np.random.seed(10)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", "--sel_dataset",   dest="sel_dataset", type=str,  help="Dataset/problem. Options=[speech_commands]")
    parser.add_argument("-attack",  "--sel_attack",    dest="sel_attack",  type=str,  help="Adv. Attack. Options=[deepfool, cw_l2, fgsm_v2, rand_pgd]")
    parser.add_argument("-method",  "--sel_method",    dest="sel_method",  type=str,  help="ACPD method. Options=[mab,mfrb,m1,m2,m3,m4,m4_1,m4_2]")

    parser.add_argument("-r",  "--root_path",          dest="root_path",        type=str, help="Root path to the folder e.g., '/../../ACPD/'")
    parser.add_argument("-lf", "--labels_file_path",   dest="labels_file_path", type=str, help="Path to the file containing the labels of the problems")
    parser.add_argument("-s",  "--save_path",          dest="save_path",        type=str, help="Path to the folder in which we want to save the results")

    parser.add_argument("-pyidx", "--p_y_obj_set_idx", dest="p_y_obj_set_idx", type=int, help="ID of the set of target distributions. 1: 100 random Dirichlets, 2: uniform distribution")

    parser.add_argument("-N_per_class",       "--N_per_class",       dest="N_per_class",       type=int, help="Number of samples per class")
    parser.add_argument("-N_per_class_train", "--N_per_class_train", dest="N_per_class_train", type=int, help="Number of samples per class for train")
    parser.add_argument("-min_multistart",    "--min_multistart",    dest="min_multistart",    type=int, help="Minimum index for multistart")
    parser.add_argument("-max_multistart",    "--max_multistart",    dest="max_multistart",    type=int, help="Maximum index for multistart")

    parser.add_argument("-maxdist_thr", "--max_dist_thr",  dest="max_dist_thr", type=float, default=None, help="Distortion threshold for the attack")
    parser.add_argument("-maxiter_thr", "--max_iter_thr",  dest="max_iter_thr", type=int,   default=None, help="Iteration  threshold for the attack")

    parser.add_argument("-probs_method", "--probs_method", dest="probs_method", type=int,   default=1,    help="Strategy to sample the probabilities from T (only =1 supported)")

    parser.add_argument("-df_ov",    "--overshoot_df", dest="overshoot_df", type=float, default=None, help="Only for Deepfool attack: overshoot parameter")

    args = parser.parse_args()

    # Root path of the project
    root_path = args.root_path

    # Load the problem labels
    labels_file_path = args.labels_file_path
    labels = open(labels_file_path, 'r').read().split('\n')

    num_classes = len(labels)  # Number of classes

    # Reach thresholds
    max_dist_thr = args.max_dist_thr
    max_iter_thr = args.max_iter_thr

    # Load the set of target probability distributions that will be tested
    p_y_obj_set_idx = int(args.p_y_obj_set_idx)
    p_y_obj_set = load_p_y_obj_set(root_path, p_y_obj_set_idx)
    #print(p_y_obj_set)


    # Initial class probabilities (uniform)
    p_y_init = [1.0/float(num_classes) for i in range(num_classes)]

    #Method to sample T
    get_row_probs_method = args.probs_method

    # General parameters
    sel_attack  = args.sel_attack   # Selected attack
    sel_dataset = args.sel_dataset  # Selected dataset
    sel_method  = args.sel_method   # Selected ACPD method
    N_per_class = args.N_per_class  # Number of samples per class

    if sel_attack == "deepfool":
        # Attack setup
        overshoot_df  = args.overshoot_df
        attack_params = [overshoot_df]
    elif sel_attack in ["rand_pgd", "cw_l2", "fgsm_v2"]:
        # Attack setup
        attack_params = []
    else:
        sys.exit("Attack not supported")


    # Load the reachability of all the samples that will be considered for the evaluation
    full_filenames,\
    full_ground_truth,\
    full_reachability = load_and_threshold_reachability(root_path, sel_attack, sel_dataset,
                                                        N_per_class, num_classes, attack_params,
                                                        max_dist_thr=max_dist_thr,
                                                        max_iter_thr=max_iter_thr)
    # Sanity check: ensure that the ground truth class is always reachable:
    for i in range(full_reachability.shape[0]):
        if full_reachability[i, full_ground_truth[i]] != 1: sys.exit("Source class not reachable!")
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    # Number of samples per class (train and validation)
    N_per_class_train = args.N_per_class_train
    N_per_class_valid = N_per_class - N_per_class_train

    samples_per_class_train = [N_per_class_train for i in range(num_classes)]
    samples_per_class_valid = [N_per_class_valid for i in range(num_classes)]

    # k-fold setup:
    min_multistart = args.min_multistart
    max_multistart = args.max_multistart
    N_multistarts  = max_multistart - min_multistart
    print("Number of k-fold multistarts: %d (from %d to %d)" % (N_multistarts, min_multistart, max_multistart))
    N_folds = N_per_class//N_per_class_train
    print("Number of folds: %d" % N_folds)

    # K-fold cross validation
    for i_start in range(min_multistart, max_multistart):

        print("Start: %d" % i_start)
        multistart_idx = multistart_shuffle(i_start, N_per_class, num_classes)

        # Sanity checks
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        tmp_init_multistart_idx = np.copy(multistart_idx)  # for sanity checks
        if len(np.unique(multistart_idx)) != (N_per_class*num_classes): sys.exit("Wrong multistart vector")
        if np.any(multistart_idx < 0): sys.exit("Invalid index: <0")
        if np.any(multistart_idx >= (N_per_class*num_classes)): sys.exit("Invalid index: >= size")
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        # Process each fold
        for i_fold in range(N_folds):

            print("Fold: %d" % i_fold)

            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            tmp_prev_multistart_idx = np.copy(multistart_idx)  # for sanity checks
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

            # ROLL THE MULTISTART VECTOR --> CHANGE THE FOLD
            if i_fold > 0:
                multistart_idx = np.roll(multistart_idx, N_per_class_train*num_classes)

            # Sanity checks
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            if i_fold == 0:
                if not np.all(tmp_init_multistart_idx == multistart_idx): sys.exit("Wrong first fold (1)")
                if not np.all(tmp_prev_multistart_idx == multistart_idx): sys.exit("Wrong first fold (2)")
            else:
                for i in range(len(multistart_idx)):
                    tmp_init_idx = (i-(N_per_class_train*num_classes*i_fold)) % (N_per_class*num_classes)
                    if multistart_idx[i] != tmp_init_multistart_idx[tmp_init_idx]: sys.exit("Wrong fold roll (1)")

                    tmp_prev_idx = (i-(N_per_class_train*num_classes)) % (N_per_class*num_classes)
                    if multistart_idx[i] != tmp_prev_multistart_idx[tmp_prev_idx]: sys.exit("Wrong fold roll (2)")
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

            # Copy the current train/validation partitions
            train_partition = np.copy(multistart_idx[0:(N_per_class_train*num_classes)])
            valid_partition = np.copy(multistart_idx[(N_per_class_train*num_classes):(N_per_class*num_classes)])

            reachability_train = np.copy(full_reachability[train_partition, :])
            reachability_valid = np.copy(full_reachability[valid_partition, :])

            ground_truth_train = np.copy(full_ground_truth[train_partition])
            ground_truth_valid = np.copy(full_ground_truth[valid_partition])

            # Sanity checks... (ensure that all are selected, no overlap, etc.)
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            if len(np.unique(train_partition)) != (N_per_class_train*num_classes): sys.exit("Reps. in train partition")
            if len(np.unique(valid_partition)) != (N_per_class_valid*num_classes): sys.exit("Reps. in valid partition")
            if len(np.unique(np.hstack((train_partition, valid_partition)))) != (N_per_class*num_classes):
                sys.exit("Reps in the merge between train and valid partitions")
            # Sanity checks: shape of reachability matrix
            if reachability_train.shape[0]     != (N_per_class_train*num_classes): sys.exit("Wrong rows in train")
            if reachability_valid.shape[0]     != (N_per_class_valid*num_classes): sys.exit("Wrong rows in valid")
            if reachability_train.shape[1]     != num_classes:                     sys.exit("Wrong cols in train")
            if reachability_valid.shape[1]     != num_classes:                     sys.exit("Wrong cols in valid")
            # Sanity checks: initial distributions
            if len(ground_truth_train)         != (N_per_class_train*num_classes): sys.exit("Wrong length in train GT")
            if len(ground_truth_valid)         != (N_per_class_valid*num_classes): sys.exit("Wrong length in valid GT")
            if np.any( [np.sum(ground_truth_train == l) != N_per_class_train for l in range(num_classes)] ):
                sys.exit("Train sampling did not maintain the initial uniform prob. distr.!")
            if np.any( [np.sum(ground_truth_valid == l) != N_per_class_valid for l in range(num_classes)] ):
                sys.exit("Valid sampling did not maintain the initial uniform prob. distr.!")
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            # Sanity check: ensure that the ground truth class is always reachable:
            for i in range(reachability_train.shape[0]):
                if reachability_train[i, ground_truth_train[i]] != 1: sys.exit("Source class not reachable (train split)!")
            for i in range(reachability_valid.shape[0]):
                if reachability_valid[i, ground_truth_valid[i]] != 1: sys.exit("Source class not reachable (valid split)!")
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


            # SUCCESS AND FOOLING RATE
            success_optimization = np.zeros(len(p_y_obj_set), dtype=int)  # success generating T in the optimization
            fooling_rates        = np.zeros(len(p_y_obj_set))  # fooling rate of the attack for the individual samples
            # KULLBACK-LEIBLER
            kl_diver_orig        = np.zeros(len(p_y_obj_set))  # kullback-leibler divergences between original & empirical
            kl_diver_obj         = np.zeros(len(p_y_obj_set))  # kullback-leibler divergence between objective & empirical
            kl_diver_orig_obj    = np.zeros(len(p_y_obj_set))  # kullback-leibler divergence between original & empirical
            # DISTANCE-BASED METRICS
            max_diff_orig        = np.zeros(len(p_y_obj_set))  # Max  diff. between original & empirical
            mean_diff_orig       = np.zeros(len(p_y_obj_set))  # Mean diff. between original & empirical
            max_diff_obj         = np.zeros(len(p_y_obj_set))  # Max  diff. between objective & empirical
            mean_diff_obj        = np.zeros(len(p_y_obj_set))  # Mean diff. between objective & empirical
            max_diff_orig_obj    = np.zeros(len(p_y_obj_set))  # Max  diff. between original & objective
            mean_diff_orig_obj   = np.zeros(len(p_y_obj_set))  # Mean diff. between original & objective
            # CORRELATIONS
            spearman_obj         = np.zeros(len(p_y_obj_set))  # Spearman correlation between objective & empirical
            pearson_obj          = np.zeros(len(p_y_obj_set))  # Pearson correlation between objective & empirical


            # ORIGINAL AND PREDICTED CLASSES
            save_orig_classes = np.zeros((len(p_y_obj_set), len(ground_truth_valid)), dtype=int)  # Ground-truth classes of the validation set
            save_adv_classes  = np.zeros((len(p_y_obj_set), len(ground_truth_valid)), dtype=int)  # Adversarial classes sampled for the validation set


            # We also compute beforehand the maximum FR that can be obtained in the 'validation' set.
            # We do it now because it only depends on the validation set, not in the objective prob. distr.
            # Since the ground-truth class is always reachable, we count the number of inputs for which we can
            # reach more than one class:
            sum_reach_valid_rowise = np.sum(reachability_valid, axis=1)  # Sum the number of classes that can be reached
            opt_fooling_rate = np.sum(sum_reach_valid_rowise > 1)/float(reachability_valid.shape[0])

            # Sanity checks
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
            if len(sum_reach_valid_rowise) != reachability_valid.shape[0]:     sys.exit("Wrong axis for sum")
            if len(sum_reach_valid_rowise) != (N_per_class_valid*num_classes): sys.exit("Wrong axis for sum")
            if np.any(sum_reach_valid_rowise < 1):             sys.exit("Zero-row in reachability_valid")
            if np.any(np.sum(reachability_train, axis=1) < 1): sys.exit("Zero-row in reachability_train")
            #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

            # COMPUTE THE AUXILIARY VARIABLES REQUIRED FOR THE SELECTED METHOD
            if sel_method in ["m1", "m2", "m3"]:
                #############
                # Compute R #
                #############
                R = np.zeros((num_classes, num_classes))
                for i in range(reachability_train.shape[0]):
                    cur_gt_label = ground_truth_train[i]
                    for target_class in range(num_classes):
                        if reachability_train[i, target_class] == 1:
                            R[cur_gt_label, target_class] += 1

                # Sanity checks:
                #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
                # Ensure that all the transitions have been computed
                if np.sum(R) != np.sum(reachability_train): sys.exit("Check computation of R")
                # Ensure that we can always "reach" the original class
                for i in range(num_classes):
                    if R[i, i] != samples_per_class_train[i]: sys.exit("Check the diagonal of R!")
                #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

            elif sel_method in ["m4", "m4_1", "m4_2"]:
                if sel_method == "m4_1":
                    m4_all_nonzero = False
                    m4_ignore_non_foolable = False
                elif sel_method == "m4_2":
                    m4_all_nonzero = True
                    m4_ignore_non_foolable = False
                elif sel_method == "m4":
                    m4_all_nonzero = True
                    m4_ignore_non_foolable = True
                else: sys.exit("Wrong string comparison??")
                subset_probs = compute_subset_probs(reachability_train, ground_truth_train, num_classes,
                                                    all_nonzero=m4_all_nonzero,
                                                    ignore_non_foolable=m4_ignore_non_foolable)
            elif sel_method in ["mab", "mfrb"]:
                print("Preprocessing not required for %s" % sel_method)

            else:
                sys.exit("Supported methods: [mab, mfrb, m1, m2, m3, m4, m4_1, m4_2]")


            # Process every p_y_obj in the set
            ##################################
            for i_p in range(len(p_y_obj_set)):
                p_y_obj = p_y_obj_set[i_p]

                start = time.time()

                if sel_method == "m1":
                    T_norm, solver_status = generate_T_agnostic(p_y_init, R, p_y_obj, num_classes)
                elif sel_method == "m2":
                    T_norm, solver_status = generate_T_method_2(p_y_init,
                                                                samples_per_class_train,
                                                                R, p_y_obj, num_classes)
                elif sel_method == "m3":
                    T_norm, solver_status = generate_T_method_3(p_y_init,
                                                                samples_per_class_train,
                                                                R, p_y_obj, num_classes)
                elif sel_method in ["m4", "m4_1", "m4_2"]:
                    T_norm, V_opt, solver_status = generate_T_method_4(p_y_init,
                                                                       p_y_obj,
                                                                       subset_probs,
                                                                       num_classes)
                elif sel_method == "mab":
                    T_norm, solver_status = generate_T_MAB(p_y_obj, num_classes)
                elif sel_method == "mfrb":
                    T_norm, solver_status = generate_T_MFRB(p_y_obj, num_classes)
                else:
                    sys.exit("Supported methods: [mab, mfrb, m1, m2, m3, m4, m4_1, m4_2]")

                end   = time.time()
                opt_solve_time = end-start

                if solver_status != "Optimal":
                    continue

                # Check that results are correct
                T_norm = np.clip(T_norm, a_min=0.0, a_max=1.0)  # clip to avoid precision-related errors

                if sel_method in ["m1", "m2", "m3", "m4", "m4_1", "m4_2"]:
                    report_failure = check_results(T_norm, p_y_init, p_y_obj, num_classes, tolerance=1e-5)
                    if report_failure: sys.exit("Failure reported in the generated T!")

                if sel_method in ["m4", "m4_1", "m4_2"]:
                    V_opt = np.clip(V_opt, a_min=0.0, a_max=1.0)  # clip to avoid precision-related errors
                    report_failure = check_V(V_opt, num_classes, tolerance=1e-5)
                    if report_failure: sys.exit("Failure reported in the generated V!")

                if sel_method in ["mab", "mfrb"]:
                    # Only the basic sanity checks for the baseline:
                    if np.any(T_norm < 0) or np.any(T_norm > 1):
                        print("-- NO! It does not satisfy $t_{i,j}<0 ^ t_{i,j}>1, forall i,j$")
                        sys.exit("Failure reported in the generated T!")
                    if np.any( np.abs( np.sum(T_norm, axis=1) - 1.0 ) > 1e-5 ):
                        print("ERROR! Not all the rows in T are probability distributions")
                        sys.exit("Failure reported in the generated T!")




                ##################
                ### Validation ###
                ##################

                ### Launch experiment ###
                #########################
                original_props,\
                adversarial_props,\
                original_classes,\
                predicted_classes = validate_T_matrix(reachability_valid,
                                                    ground_truth_valid,
                                                    get_row_probs_method,
                                                    T_norm,  num_classes)


                # SUCCESS AND FOOLING RATE
                success_optimization[i_p] = 1
                fooling_rates[i_p] = np.sum(original_classes != predicted_classes)/len(original_classes)
                # DISTANCE-BASED METRICS
                max_diff_orig[i_p]      = np.max( np.abs( np.array(original_props) - np.array(adversarial_props) ))
                mean_diff_orig[i_p]     = np.mean(np.abs( np.array(original_props) - np.array(adversarial_props) ))
                max_diff_obj[i_p]       = np.max( np.abs( np.array(p_y_obj)  - np.array(adversarial_props) ))
                mean_diff_obj[i_p]      = np.mean(np.abs( np.array(p_y_obj)  - np.array(adversarial_props) ))
                max_diff_orig_obj[i_p]  = np.max( np.abs( np.array(p_y_obj)  - np.array(original_props) ))
                mean_diff_orig_obj[i_p] = np.mean(np.abs( np.array(p_y_obj)  - np.array(original_props) ))
                # CORRELATIONS
                spearman_obj[i_p] = sp.stats.spearmanr(adversarial_props, p_y_obj)[0]
                pearson_obj[i_p]  = sp.stats.pearsonr(adversarial_props, p_y_obj)[0]
                # KULLBACK-LEIBLER
                kl_diver_orig[i_p]     = entropy(original_props, adversarial_props)
                kl_diver_obj[i_p]      = entropy(p_y_obj,        adversarial_props)
                kl_diver_orig_obj[i_p] = entropy(original_props, p_y_obj)
                # Sanity check: avoid INF values of the KL because of zeros in the produced distribution
                if np.isinf(kl_diver_orig[i_p]) or np.isinf(kl_diver_obj[i_p]):
                    if np.any(adversarial_props == 0.0):
                        # Laplacian smoothing
                        sm_adv_props = np.copy(adversarial_props)
                        sm_adv_props = np.array([ sm_adv_props[i]*(N_per_class_valid*num_classes)+1 for i in range(num_classes) ])
                        sm_adv_props = np.array([ sm_adv_props[i]/np.sum(sm_adv_props) for i in range(num_classes) ])
                        if np.abs(np.sum(sm_adv_props)-1.0) > 1e-8: sys.exit("Check Laplacian smoothing")
                        kl_diver_orig[i_p] = entropy(original_props, sm_adv_props)
                        kl_diver_obj[i_p]  = entropy(p_y_obj,        sm_adv_props)
                        if np.isinf(kl_diver_orig[i_p]) or np.isinf(kl_diver_obj[i_p]): sys.exit("Check KL computation")
                    else: sys.exit("Check KL computation (inf/nan but not zeros)")
                # ORIGINAL / ADVERSARIAL CLASSES
                save_orig_classes[i_p,:] = np.copy(original_classes)
                save_adv_classes[i_p, :] = np.copy(predicted_classes)

            # end i_p (prob. distributions)

            save_path = args.save_path
            print("Saving in %s" % save_path)

            if sel_attack == "deepfool":
                param_appendix = "_ov_%s_eps_%s_iters_%d_start_%d_fold_%d.npy"%(str(overshoot_df),
                                                                                str(max_dist_thr),
                                                                                max_iter_thr,
                                                                                i_start, i_fold)
            elif sel_attack in ["rand_pgd", "cw_l2", "fgsm_v2"]:
                param_appendix = "_eps_%s_iters_%d_start_%d_fold_%d.npy" % (str(max_dist_thr), max_iter_thr, i_start, i_fold)
            else:
                sys.exit("Attack not supported")


            # SUCCESS AND FOOLING RATE
            np.save(save_path + "success_opt"       + param_appendix, success_optimization)
            np.save(save_path + "fr"                + param_appendix, fooling_rates)
            np.save(save_path + "opt_fr"            + param_appendix, opt_fooling_rate)
            # KULLBACK-LEIBLER
            np.save(save_path + "kl_orig"           + param_appendix, kl_diver_orig)
            np.save(save_path + "kl_obj"            + param_appendix, kl_diver_obj)
            np.save(save_path + "kl_orig_obj"       + param_appendix, kl_diver_orig_obj)
            # DISTANCE BASED METRICS
            np.save(save_path + "max_dif_orig"      + param_appendix, max_diff_orig)
            np.save(save_path + "mean_dif_orig"     + param_appendix, mean_diff_orig)
            np.save(save_path + "max_dif_obj"       + param_appendix, max_diff_obj)
            np.save(save_path + "mean_dif_obj"      + param_appendix, mean_diff_obj)
            np.save(save_path + "max_dif_orig_obj"  + param_appendix, max_diff_orig_obj)
            np.save(save_path + "mean_dif_orig_obj" + param_appendix, mean_diff_orig_obj)
            # CORRELATIONS
            np.save(save_path + "spearman_obj"      + param_appendix, spearman_obj)
            np.save(save_path + "pearson_obj"       + param_appendix, pearson_obj)

            # SAVED ORIGINAL / ADVERSARIAL CLASSES
            np.save(save_path + "save_orig_classes" + param_appendix, save_orig_classes)
            np.save(save_path + "save_adv_classes"  + param_appendix, save_adv_classes)

            print(np.sum(success_optimization))
            print(np.mean(fooling_rates))
            print(np.mean(kl_diver_obj))
            print(np.mean(max_diff_obj))
        # end i_fold

    # end i_multistart

    print("Job done!")
