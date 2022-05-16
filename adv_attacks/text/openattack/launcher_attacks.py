import os
import os.path
import sys

from oa_utils import text_model_tag2id

if __name__ == '__main__':

    base_path = "../../../"

    # LOAD LABELS
    ####################
    main_setup = "emotion"
    print("Main setup: \t " + main_setup)

    # TARGET MODEL PATH
    ####################
    #Emotion tags (from bhadresh-savani - HuggingFace repository)
    #model_tag = "roberta-base-emotion"
    model_tag = "bert-base-uncased-emotion" # ** Model selected in the paper
    #model_tag = "distilbert-base-uncased-emotion"
    #model_tag = "albert-base-v2-emotion"
    model_id  = text_model_tag2id(main_setup, model_tag, base_path)  # model ID (abbreviated identifier)
    print("Model tag: %s (id: %s)" % (model_tag, model_id))

    model_path = base_path + "models/%s" % model_tag
    print("Path of the target model: \t " + model_path)

    # DATASET PATH
    ####################
    dataset_path = base_path + "datasets/%s_dataset_train" % main_setup
    print("Dataset path: %s" % dataset_path)

    # ATTACK CONFIGURATION
    ######################
    attack_type  = "genetic"  # Selected adversarial attack
    target_classes = [0, 1, 2, 3, 4, 5]

    # INPUTS TO BE PROCESSED
    ########################
    input_filename = base_path + "adv_attacks/text/experiments/%s_%s_full_sampling.npy" % (main_setup, model_id)

    # LAUNCH
    #############
    print("Attack type:", attack_type)

    for target_class in target_classes:
        print("Target class:", target_class)

        save_path = base_path + "adv_attacks/text/%s/" % attack_type
        if target_class is None:
            save_path = save_path + "results_untarg/"
        else:
            save_path = save_path + "results_targ/"

        cmd = "python3 Attacks.py"
        cmd = cmd + " --main_setup " + main_setup
        cmd = cmd + " --dataset_path " + dataset_path
        cmd = cmd + " --model_path " + model_path
        cmd = cmd + " --model_id " + model_id
        cmd = cmd + " --save_path " + save_path
        cmd = cmd + " --attack_type " + attack_type
        cmd = cmd + " --input_filename " + input_filename
        if not (target_class is None):
            cmd = cmd + " --target_class " + str(target_class)

        # Launch process
        print(cmd)
        os.system(cmd)

    print("All processes have been successfully launched")
