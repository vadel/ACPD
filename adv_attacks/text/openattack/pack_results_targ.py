import os

import argparse
import sys
import numpy as np
import re

import OpenAttack as oa

import datasets  # use the Hugging Face's datasets library
datasets.set_caching_enabled(False)  # DISABLE CACHES IN THE DATASET PROCESSING!
datasets.config.IN_MEMORY_MAX_SIZE = 2**30  # 1GB

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-setup", "--main_setup",     dest="main_setup",     help="Main setup [emotion]")
    parser.add_argument("-d",     "--dataset_path",   dest="dataset_path",   help="Path to the dataset folder")
    parser.add_argument("-l",     "--labels_path",    dest="labels_path",    help="Path to the file containing the labels")
    parser.add_argument("-m",     "--model_path",     dest="model_path",     help="Path to the model")
    parser.add_argument("-mid",   "--model_id",       dest="model_id",       help="Model ID (abbreviated identifier)")
    parser.add_argument("-a",     "--adv_attack",     dest="adv_attack",     help="Attack type")
    parser.add_argument("-i",     "--input_filename", dest="input_filename", help="File containing the dataset indices that want to be considered")
    parser.add_argument("-r",     "--results_root",   dest="results_root",   help="Path containing the results of the adv. attacks")
    parser.add_argument("-s",     "--save_path",      dest="save_path",      help="Path to the folder in which we want to save the results")
    args = parser.parse_args()

    # Main setup identifier
    main_setup = args.main_setup

    # Read labels
    labels_file_path = args.labels_path
    labels = open(labels_file_path, 'r').readlines()
    labels = [s.rstrip() for s in labels]

    num_classes = len(labels)

    # Dataset
    dataset_path = args.dataset_path
    dataset      = datasets.load_from_disk(dataset_path, keep_in_memory=True)  # Load data from disk

    # Load the model
    MODEL     = args.model_path
    model_id  = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(MODEL)  # Load the tokenizer
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)  # Load the pretrained classifier

    # Convert the model into a transformers model
    transmodel = oa.classifiers.TransformersClassifier(model, tokenizer,
                                                       embedding_layer=model.base_model.embeddings.word_embeddings,
                                                       max_length=512)

    # Load audios to process
    input_sampling_path = args.input_filename
    full_samp = np.load(input_sampling_path)

    N_total  = len(full_samp)    # Total number of samples

    adv_attack = args.adv_attack  # Adversarial attack type selected
    print("Processing adv. attack: %s" % adv_attack)

    results_root = args.results_root
    print("Loading from %s" % results_root)

    # Load a distance metric
    metric_lev = oa.metric.Levenshtein(transmodel.tokenizer)

    # Some setup configuration
    if adv_attack in ["viper", "pwws", "textbugger", "pso", "genetic", "hotflip", "wordbug", "textfooler"]:
        def dist_metric(input, adv):
            out = metric_lev.calc_score(input, adv)
            out = out / max(len(input), len(adv))  # normalize
            return out
    else:
        print(adv_attack)
        sys.exit("Attack not supported %s" % adv_attack)

    reachability  = np.zeros((N_total, num_classes), dtype=int)  # 1 if reached, 0 else
    num_steps_vec = np.zeros((N_total, num_classes), dtype=int)  # number of model queries
    dist_vec      = np.zeros((N_total, num_classes))  # "perturbation magnitude"
    secs_vec      = np.zeros((N_total, num_classes))  # seconds required to create the adv. ex.

    for target_class, label in enumerate(labels):
        # Load the results
        filename = "%s_%s_attacks_targ_%d.npy" % (main_setup, model_id, target_class)
        result_file = results_root + filename
        print("Loading results from: ", result_file)
        results = np.load(result_file, allow_pickle=True)[()]  # [()] is because np saves/loads dicts as pickle objects
        # Sanity checks
        assert len(results.keys()) == len(full_samp)
        assert N_total == len(full_samp)
        assert not np.any([results[x]["metrics"]["Query Exceeded"] for x in results.keys()]), "Queries exceeded --> check"

        for i_f in range(N_total):
            if i_f % 100 == 0: print(i_f)

            cur_index    = int(results[i_f]["index"])
            cur_input    = results[i_f]["x"]
            cur_gt_label = results[i_f]["y"]
            assert cur_gt_label == dataset[cur_index]["y"], "Label mismatch, index %d (%d)" % (cur_index, i_f)

            if target_class == cur_gt_label:
                reachability[i_f, target_class]  = 1  # We assume that the source class is always reachable
                num_steps_vec[i_f, target_class] = 0  # number of model queries required (we assume it is zero)
                dist_vec[i_f, target_class] = 0.0     # distance(we assume it is zero)
                secs_vec[i_f, target_class] = 0.0     # time required to create the attack (we assume it is zero)
            else:
                y_adv       = results[i_f]["y_adv"]  # Predicted class
                adv_example = results[i_f]["x_adv"]  # Adversarial example
                n_steps     = results[i_f]["metrics"]["Victim Model Queries"]  # Number of model queries required
                secs        = results[i_f]["metrics"]["Running Time"]  # Time required

                if adv_example is None:
                    # Adversarial example not found
                    reachability[i_f, target_class]  = 0  # Target class not reached
                    num_steps_vec[i_f, target_class] = 0  # number of steps required (we assume it is zero)
                    dist_vec[i_f, target_class] = 0.0     # distance(we assume it is zero)
                    secs_vec[i_f, target_class] = np.copy(secs)  # we save the time anyways...
                    assert y_adv == cur_gt_label, "Label mismatch when not fooled, index %d (%d)" % (cur_index, i_f)

                else:
                    # Check if the audio has reached the target class
                    if y_adv == target_class:
                        reachability[i_f, target_class] = 1  # Reached
                    else:
                        reachability[i_f, target_class] = 0  # Not reached
                    num_steps_vec[i_f, target_class] = np.copy(n_steps)  # number of steps required
                    dist_vec[i_f, target_class] = dist_metric(cur_input, adv_example)  # Save the 'distortion level'
                    secs_vec[i_f, target_class] = np.copy(secs)  # Save the time required to create the attack

    # Sanity checks
    assert not np.any(np.isnan(dist_vec)), "Nans in distance metrics"
    assert not np.any(np.isnan(num_steps_vec)), "Nans in number of steps"
    assert not np.any(np.isnan(secs_vec)), "Nans in time measurements"
    assert not np.any(np.isinf(dist_vec)), "Infs in distance metrics"
    assert not np.any(np.isinf(num_steps_vec)), "Infs in number of steps"
    assert not np.any(np.isinf(secs_vec)), "Infs in time measurements"


    # Save the results
    save_path = args.save_path
    print("Saving results in %s" % save_path)

    appendix_param_npy = "%s_targ.npy" % adv_attack

    np.save(save_path + "reachability_"  + appendix_param_npy, reachability)
    np.save(save_path + "num_steps_vec_" + appendix_param_npy, num_steps_vec)
    np.save(save_path + "dist_vec_" + appendix_param_npy, dist_vec)
    np.save(save_path + "secs_vec_" + appendix_param_npy, secs_vec)

    print("Job done!")
