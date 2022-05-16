from scipy.special import softmax
import numpy as np

def classify(text, model, tokenizer, preprocess, device):
    prep_text = preprocess(text)

    encoded_input = tokenizer(prep_text, return_tensors='pt').to(device)
    output = model(**encoded_input)
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)

    return scores


def mapping_preprocess(x, preprocess=None):
    if preprocess is not None: x["x"] = preprocess(x["x"])
    return x


def delete_target_feature(dataset):
    return dataset.map(remove_columns=["target"], keep_in_memory=True, load_from_cache_file=False)


def set_target_to_dataset(dataset, target_class):
    return dataset.map(function=lambda x: {"target": target_class}, keep_in_memory=True, load_from_cache_file=False)


def text_model_tag2id(main_setup, model_tag, base_path="../../../"):
    load_filename = base_path + "models/" + "%s_tag2id_dict.npy" % main_setup  # Filename
    model_tag2id  = np.load(load_filename, allow_pickle=True)[()]  # Load the dictionary
    assert model_tag in model_tag2id, "Tag not found in the dictionary: %s" % model_tag
    return model_tag2id[model_tag]


def text_model_id2tag(main_setup, model_id, base_path="../../../"):
    load_filename = base_path + "models/" + "%s_tag2id_dict.npy" % main_setup  # Filename
    model_tag2id  = np.load(load_filename, allow_pickle=True)[()]  # Load the dictionary

    model_id2tag  = {v_: k_ for k_, v_ in model_tag2id.items()}  # Revert the dictionary

    assert model_id in model_id2tag, "ID not found in the dictionary: %s" % model_id
    return model_id2tag[model_id]
