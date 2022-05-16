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
    model_tag = "bert-base-uncased-emotion"  # ** Model selected in the paper
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

    # SAVE PATH
    ####################
    save_path = base_path + "adv_attacks/text/analysis/predictions/"

    # LAUNCH
    ####################
    cmd = "python3 Classify_train.py"
    cmd = cmd + " --main_setup "   + main_setup
    cmd = cmd + " --dataset_path " + dataset_path
    cmd = cmd + " --model_path "   + model_path
    cmd = cmd + " --model_id "     + model_id
    cmd = cmd + " --save_path "    + save_path

    # Launch process
    print(cmd)
    os.system(cmd)

    print("All processes have been successfully launched")
