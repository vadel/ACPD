#!/usr/bin/env python3
import argparse
import datasets  # use the Hugging Face's datasets library
datasets.set_caching_enabled(False)  # DISABLE CACHES IN THE DATASET PROCESSING!
datasets.config.IN_MEMORY_MAX_SIZE = 2**30 #1GB

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
#from transformers import pipeline
import numpy as np
import sys

from oa_utils import classify

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-setup", "--main_setup",   dest="main_setup",   help="Main setup [emotion]")
    parser.add_argument("-d",     "--dataset_path", dest="dataset_path", help="Path to the dataset folder")
    parser.add_argument("-m",     "--model_path",   dest="model_path",   help="Path to the model")
    parser.add_argument("-mid",   "--model_id",     dest="model_id",     help="Model ID (abbreviated identifier")
    parser.add_argument("-s",     "--save_path",    dest="save_path",    help="Path to the folder in which the results will be saved")
    args = parser.parse_args()

    main_setup   = args.main_setup    # Setup option
    dataset_path = args.dataset_path  # Dataset path in disk
    dataset = datasets.load_from_disk(dataset_path, keep_in_memory=True)  # Load data from disk

    if main_setup == "emotion":
        # Label mapping
        labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        # 1) Preprocess function
        def preprocess(text): return text
        # 2) Dataset adaptation functions
        def dataset_mapping(x): return {"x": x["text"], "y": x["label"]}

    else:
        sys.exit("Supported options: ['emotion']")

    MODEL    = args.model_path # Model path (downloaded from HuggingFaces)
    model_id = args.model_id   # Model ID (abbreviated identifier) - defined in "ACPD/models/emotion_tag2id_dict.npy

    tokenizer = AutoTokenizer.from_pretrained(MODEL)  # Load the tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)  # Load the pretrained classifier

    n_rows_train = dataset.num_rows
    assert n_rows_train == len(dataset)       # Sanity check
    assert n_rows_train == len(dataset["y"])  # Sanity check

    pred_labels_train = np.zeros(n_rows_train, dtype=np.int16)
    for i in range(n_rows_train):
        if i % 5000 == 0: print(i)
        cur_scores = classify(dataset[i]["x"], model, tokenizer, preprocess, device)
        cur_class = np.argmax(cur_scores)
        pred_labels_train[i] = np.copy(cur_class)

    acc_train = np.sum(pred_labels_train == np.array(dataset["y"], dtype=np.int16)) / n_rows_train * 100
    print(acc_train)

    #Save results
    save_path = args.save_path
    print("Saving results in %s" % save_path)

    #Save the model predictions
    filename = save_path + "%s_%s_predictions_train.npy" % (main_setup, model_id)
    np.save(filename, pred_labels_train)

