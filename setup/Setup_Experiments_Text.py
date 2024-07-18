#!/usr/bin/env python
import torch
import datasets # use the Hugging Face's datasets library
datasets.set_caching_enabled(False)  # disable caches in the dataset processing
datasets.config.IN_MEMORY_MAX_SIZE = 2**30
import numpy as np
import csv
import urllib.request
import copy 
import sys
np.random.seed(111)
torch.manual_seed(111)
torch.random.manual_seed(111)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Select the main setup / dataset
base_path = "../"
main_setup = "emotion"
print("Selected setup/dataset: %s"%main_setup)

# Load the configuration for the selected setup
def mapping_preprocess(x, preprocess=None):
    if preprocess is not None: x["x"] = preprocess(x["x"])
    return x

if main_setup == "emotion":
    # download label mapping
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    print("Labels:", labels)
    #Load the dataset
    def preprocess(text): return text
    #2) Dataset adaptation functions
    def dataset_mapping(x): return { "x": x["text"], "y": x["label"]}
    #Load data
    #dataset = datasets.load_dataset("emotion", split=None, keep_in_memory=True, download_mode='force_redownload')
    dataset = datasets.load_dataset("dair-ai/emotion", split=None, keep_in_memory=True, download_mode='force_redownload')
else:
    sys.exit("Main setup not supported yet. Options: [emotion]")

#Convert "text"/"label" into "x"/"y"
dataset = dataset.map(function=dataset_mapping, keep_in_memory=True, load_from_cache_file=False)
#Preprocess the text ("x")
dataset = dataset.map(function=mapping_preprocess, fn_kwargs={"preprocess":preprocess}, keep_in_memory=True, load_from_cache_file=False)

#Save the training dataset in a more accessible format:
dataset["train"].save_to_disk(base_path + "datasets/%s_dataset_train" % main_setup)
print("Dataset saved!")

#Save a file with the labels
label_file_path = base_path + "datasets/%s_dataset_train/%s_labels.txt" % (main_setup, main_setup)
with open(label_file_path, 'w') as f:
    for lab in labels:
        f.write("%s\n" % lab)

# Create and save a dictionary that maps the model tags and the identifiers (from bhadresh-savani - HuggingFace repository)
if main_setup == "emotion":
    model_tag2id = {"roberta-base-emotion": "m0",
                    "bert-base-uncased-emotion": "m1",
                    "distilbert-base-uncased-emotion": "m2",
                    "albert-base-v2-emotion": "m3"}

else:
    sys.exit("Main setup not supported yet. Options: [emotion]")

# Save the dictionary
np.save(base_path + "models/%s_tag2id_dict.npy" % main_setup, model_tag2id)


