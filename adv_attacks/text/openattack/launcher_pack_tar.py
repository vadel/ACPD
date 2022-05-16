import os
import os.path
import sys

from oa_utils import text_model_tag2id

if __name__ == '__main__':

    base_path = "../../../"

    # MAIN SETUP
    ####################
    main_setup = "emotion"
    print("Main setup: %s" % main_setup)

    # LABEL PATH
    ####################
    labels_path = base_path + "datasets/%s_dataset_train/%s_labels.txt" % (main_setup, main_setup)
    print("File containing the labels: %s" % labels_path)

    # ADVERSARIAL ATTACK TO PROCESS
    ####################
    adv_attack = "genetic"


    # TARGET MODEL PATH
    ####################
    #Emotion tags (from bhadresh-savani - HuggingFace repository)
    #model_tag = "roberta-base-emotion"
    model_tag = "bert-base-uncased-emotion"  # ** model used in the paper
    #model_tag = "distilbert-base-uncased-emotion"
    #model_tag = "albert-base-v2-emotion"
    model_id  = text_model_tag2id(main_setup, model_tag, base_path)  # model ID (abbreviated identifier)
    print("Model tag: %s (id: %s)" % (model_tag, model_id))

    model_path = base_path + "models/%s" % model_tag
    print("Path of the target model: %s" % model_path)

    # DATASET PATH
    ####################
    dataset_path = base_path + "datasets/%s_dataset_train/" % main_setup
    print("Dataset path: %s" % dataset_path)

    # INPUTS TO BE PROCESSED
    ########################
    input_filename = base_path + "adv_attacks/text/experiments/%s_%s_full_sampling.npy" % (main_setup, model_id)
    print("Inputs to be processed: %s" % input_filename)

    # SAVE PATH
    ####################
    save_path = base_path + "adv_attacks/text/analysis/targeted/%s/%s/" % (main_setup, model_id)
    print("Results will be saved at %s" % save_path)

    print("Selected attack: %s" % adv_attack)

    # WHERE TO LOAD THE RESULTS OF THE ADVERSARIAL ATTACKS
    ########################
    results_root = base_path + "adv_attacks/text/%s/results_targ/" % adv_attack
    print("Results will be loaded from: %s" % results_root)

    # LAUNCH
    ####################
    cmd = "python3 pack_results_targ.py"
    cmd = cmd + " --main_setup " + main_setup
    cmd = cmd + " --dataset_path " + dataset_path
    cmd = cmd + " --labels_path " + labels_path
    cmd = cmd + " --model_path " + model_path
    cmd = cmd + " --model_id " + model_id
    cmd = cmd + " --adv_attack " + adv_attack
    cmd = cmd + " --input_filename " + input_filename
    cmd = cmd + " --results_root " + results_root
    cmd = cmd + " --save_path " + save_path

    # Launch process
    print(cmd)
    os.system(cmd)

    print("All processes have been successfully launched")
