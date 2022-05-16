#!/usr/bin/env python3
import argparse

import OpenAttack as oa
import datasets  # use the Hugging Face's datasets library
datasets.set_caching_enabled(False)  # DISABLE CACHES IN THE DATASET PROCESSING!
datasets.config.IN_MEMORY_MAX_SIZE = 2**30  # 1GB

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import sys
import random

from oa_utils import classify
from oa_utils import delete_target_feature
from oa_utils import mapping_preprocess
from oa_utils import set_target_to_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.random.manual_seed(111)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-setup", "--main_setup",   dest="main_setup",   help="Main setup. Options: [emotion]")
    parser.add_argument("-d",     "--dataset_path", dest="dataset_path", help="Path to the dataset folder")
    parser.add_argument("-m",     "--model_path",   dest="model_path",   help="Path to the model")
    parser.add_argument("-mid",   "--model_id",     dest="model_id",     help="Model ID (abbreviated identifier")
    parser.add_argument("-s",     "--save_path",    dest="save_path",    help="Path to the folder in which the results will be saved")
    parser.add_argument("-a",     "--attack_type",  dest="attack_type",  help="Attack type [genetic, hotflip, wordbug, pwws, viper, textbugger, textfooler, pso]")
    parser.add_argument("-t",     "--target_class", dest="target_class", type=int, default=None,
                        help="Target class (if not specified, untargeted attacks will be launched")
    parser.add_argument("-i", "--input_filename", dest="input_filename", help="File containing the dataset indices that want to be considered")
    args = parser.parse_args()

    main_setup = args.main_setup
    dataset_path = args.dataset_path  # Dataset path in disk
    dataset = datasets.load_from_disk(dataset_path, keep_in_memory=True)  # Load data from disk

    if main_setup == "emotion":
        # Labels
        labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        # 1) Preprocess function
        def preprocess(text): return text
        # 2) Dataset adaptation functions
        def dataset_mapping(x): return {"x": x["text"], "y": x["label"]}

    else:
        sys.exit("Supported options: ['emotion']")

    MODEL = args.model_path  # Model path (downloaded from HuggingFaces)

    model_id = args.model_id  # Model ID (abbreviated identifier) - defined in "ACPD/models/emotion_tag2id_dict.npy"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)  # Load the tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)  # Load the pretrained classifier

    # Convert the model into a transformers model
    transmodel = oa.classifiers.TransformersClassifier(model, tokenizer,
                                                       embedding_layer=model.base_model.embeddings.word_embeddings,
                                                       max_length=512)
    print("Transformer model loaded")

    # Set target class (None for untargeted attacks)
    target_class = args.target_class
    print("Selected target class:", target_class)
    if target_class is not None:
        dataset = set_target_to_dataset(dataset, target_class)
        # Sanity check
        for tc in dataset["target"]: assert tc == target_class

    # Load the indices of the inputs that we will consider in the experiments
    input_indices_file = args.input_filename
    print("Loading input indices from %s" % input_indices_file)
    full_samp = np.load(input_indices_file)  # list of indices

    # Configure the attack to be used
    attack_type = args.attack_type
    if   attack_type == "pwws":        attacker = oa.attackers.PWWSAttacker()
    elif attack_type == "textbugger":  attacker = oa.attackers.TextBuggerAttacker()
    elif attack_type == "textfooler":  attacker = oa.attackers.TextFoolerAttacker()
    elif attack_type == "viper":       attacker = oa.attackers.VIPERAttacker(generations=500)
    elif attack_type == "pso":         attacker = oa.attackers.PSOAttacker(max_iters=100)
    elif attack_type == "genetic":     attacker = oa.attackers.GeneticAttacker(max_iters=100)
    elif attack_type == "hotflip":     attacker = oa.attackers.HotFlipAttacker()
    elif attack_type == "wordbug":     attacker = oa.attackers.DeepWordBugAttacker()
    else: sys.exit("Attack not supported yet")
    print("Attacker configured")

    # Prepare for attacking
    attack_eval = oa.AttackEval(attacker, transmodel)

    # Launch the attacks iteratively
    attack_iter = attack_eval.ieval(dataset.select(full_samp, keep_in_memory=True))

    cont = 0
    result_dict = {}  # dictionary to save the results

    for cur_res in attack_iter:
        print("\n")
        print(cur_res['data']['x'])
        print(cur_res['result'])
        print(cur_res['success'])
        print(cur_res['metrics'])
        #Sanity check
        if not (target_class is None):
            assert cur_res['data']['target'] == target_class, "cur_res[data][target]=%d, target=%d" % (cur_res['data']['target'], target_class)

        clean_input  = cur_res['data']['x']
        clean_scores = classify(clean_input, model, tokenizer, preprocess, device)
        clean_label  = np.argmax(clean_scores)
        assert int(clean_label) == int(cur_res['data']['y']), "Wrongly classified input! - cont=%d, idx=%d" % (cont, full_samp[cont])
        print(clean_input)
        print("Clean prediction - True label: %d (%s)" % (int(clean_label), labels[clean_label]))

        print("Adversarial prediction")
        if not (cur_res['result'] is None):
            assert cur_res['success'], "The attack did not succeed - cont=%d, idx=%d" % (cont, full_samp[cont])  # Sanity check
            # Retrieve and classify the adversarial example
            adv_example = cur_res['result']
            print(adv_example)
            adv_scores = classify(adv_example, model, tokenizer, preprocess, device)
            adv_label  = np.argmax(adv_scores)
            print("Adversarial prediction: %d (%s)" % (int(adv_label), labels[adv_label]))
            # Sanity checks
            if target_class is None:
                assert adv_label != clean_label, "not fooled: cont=%d, idx=%d" % (cont, full_samp[cont])
            else:
                assert adv_label == target_class, "target not reached: cont=%d, idx=%d" % (cont, full_samp[cont])

        else:
            assert not cur_res['success'], "The attack succeeded?"  # Sanity check
            adv_label = clean_label  # set the clean label
            print("Adversarial example not found")

        result_dict[cont] = {"x": cur_res['data']['x'], "x_adv": cur_res['result'],
                             "y": cur_res['data']['y'], "y_adv": adv_label, "index": full_samp[cont],
                             "metrics": cur_res['metrics'], "success": cur_res['success'],
                             "text": cur_res['data']['text'], "label": cur_res['data']['label']}
        cont += 1

    # Save results
    save_path = args.save_path
    print("Saving results in %s" % save_path)

    if target_class is None: params_in_filename = "untarg"
    else:                    params_in_filename = "targ_%d" % target_class

    # Save the model predictions
    filename = save_path + "%s_%s_attacks_%s.npy" % (main_setup, model_id, params_in_filename)
    np.save(filename, result_dict)

    print("Job done!")
