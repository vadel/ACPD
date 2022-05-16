#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join

import argparse
import os.path
import sys
import struct
import time
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import random
import re


# Helper function to reset the graph and set a random seed
def reset_graph(seed=1996):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    config = tf.ConfigProto(device_count = {'GPU': 0})
#reset_graph()


# Load frozen graph
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


# Projection operator: Projects on the lp ball centered at 0 and of radius xi
def proj_lp(v, xi, p):
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


# Attack class
class Attack:
    def __init__(self, model_path, labels_file_path, num_classes=12):
        print("\t+ Loading graph...")
        self.graph = load_graph(model_path)
        self.sess = tf.Session(graph=self.graph)
        print("\t- Graph loaded!")

        self.labels = open(labels_file_path,'r').read().split('\n')
        self.num_classes = num_classes
        
        print("\t+ Restoring tensors from the model...")

        self.input_layer_name   = "prefix/input_audio:0"    #INPUT TENSOR NAME
        self.logits_layer_name  = "prefix/add_3:0"          #LOGITS TENSOR NAME
        self.softmax_layer_name = "prefix/labels_softmax:0" #SOFTMAX TENSOR NAME

        self.input_tensor   = self.graph.get_tensor_by_name(self.input_layer_name)   #LOGITS TENSOR
        self.logits_tensor  = self.graph.get_tensor_by_name(self.logits_layer_name)  #LOGITS TENSOR
        self.softmax_tensor = self.graph.get_tensor_by_name(self.softmax_layer_name) #SOFTMAX TENSOR


        self.y_flat = tf.reshape(self.logits_tensor, (-1,))

        with self.graph.as_default():
            self.inds = tf.placeholder(tf.int32, shape=(self.num_classes,))
            self.dydx = self.jacobian(self.y_flat, self.input_tensor, self.inds)
        print("\t- Tensors restored! Initialization sufccesfully completed!")

    #Feedforward function (input: input signal, output: logits).
    def f(self, audio_sample): 
        output = self.sess.run(self.logits_tensor,  feed_dict={self.input_tensor: audio_sample})
        return output
    
    #Helper function to compute Jacobians for each class
    def jacobian(self, y, x, inds):
        n = self.num_classes
        loop_vars = [
             tf.constant(0, tf.int32),
             tf.TensorArray(tf.float32, size=n),
        ]
        _, jacobian = tf.while_loop(
            lambda j,_: j < n,
            lambda j,result: (j+1, result.write(j, tf.gradients(y[inds[j]], x))),
            loop_vars)
        return jacobian.stack()

    # Helper function to compute the gradients
    def grads(self, audio_sample, indices): 
        return self.sess.run(self.dydx, feed_dict={self.input_tensor: audio_sample, self.inds: indices}).squeeze(axis=1)
    

    # Targeted DeepFool algorithm
    def targeted_deepfool(self, audio, label, target_class=None, num_classes=12, 
                        overshoot=0.02, max_iter=50, min_iter=0, verbose=False):
        """
        :param audio: audio of size [1,16000]
        :param label: ground-truth label of the audio
        :param target_class: target class of the adversarial attack
        :param num_classes: number of classes
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter: maximum number of iterations for deepfool 
        :param min_iter: minimum number of iterations for deepfool 
        :return: 
            :r_tot   : total perturbation that fools the classifier (raw, without rounding constraints)
            :adv_pert: final perturbation that fools the classifier (considering  rounding constraints)
            :loop_i  : final step index
            :k_i     : new predicted label
            :pert_audio: final adversarial example
        """

        min_ = -1.0 #Minimum value for the inputs
        max_ =  1.0 #Maximum value for the inputs

        #Logits of the initial audio
        f_audio = np.array(self.f(audio)).flatten()
        #List containing the labels different to the original one
        I = np.arange(num_classes)
        residual_labels = [l for l in I if l!=label]

        input_shape = audio.shape   #Shape of the input
        pert_audio = np.copy(audio) #Variable for the adv. example

        f_i = np.array(self.f(pert_audio)).flatten() #Logits of the adv. example
        k_i = int(np.argmax(f_i)) #Prediction for the adv. example

        w = np.zeros(input_shape)     #Variable to store the gradients
        r_tot = np.zeros(input_shape) #Variable to store the total perturbation

        loop_i = 0 #Step index

        while (k_i != target_class and loop_i < max_iter) or loop_i < min_iter:
            
            pert = np.inf
            gradients = np.asarray(self.grads(pert_audio, I)) #Compute gradients

            #For TARGETED attacks, move directly towards the target class
            k = target_class
            w_k = gradients[k, :, :] - gradients[label, :, :]
            f_k = f_i[k] - f_i[label]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            pert = pert_k
            w = w_k

            #Compute r_i and r_tot
            r_i = pert * w / np.linalg.norm(w)
            r_tot = r_tot + r_i

            # Compute new perturbed input
            pert_audio = audio + (1+overshoot)*r_tot
            pert_audio = np.clip(pert_audio, min_, max_)

            # Ensure that the input satisfies format restrictions
            pert_audio = np.array(np.clip(np.round(pert_audio*(1<<15)), -2**15, 2**15-1),dtype=np.int16)
            pert_audio = pert_audio/(1<<15)

            # Compute new logits and label
            f_i = np.array(self.f(pert_audio)).flatten()
            k_i = int(np.argmax(f_i))

            loop_i += 1

        r_tot = (1+overshoot)*r_tot #Final total (raw) perturbation
        adv_pert = pert_audio.flatten() - audio.flatten() #Final perturbation considering restrictions

        return r_tot, adv_pert, loop_i, k_i, pert_audio





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--input_filenames",   dest="input_files_path",  help="Full path to the numpy file (.npy) containing input audio filenames")
    parser.add_argument("-minidx", "--minidx",        dest="minidx",            help="Indicates the index of the first audio to process (from --input filenames)", type=int, default=-1)
    parser.add_argument("-maxidx", "--maxidx",        dest="maxidx",            help="Indicates the index of the last  audio to process (from --input filenames)", type=int, default=-1)

    parser.add_argument("-l",  "--label_idx",         dest="label_idx",         help="Integer representing the label index", type=int)
    parser.add_argument("-m",  "--model_path",        dest="model_path",        help="Path to the frozen TF model (.pb)")
    parser.add_argument("-lf", "--labels_file_path",  dest="labels_file_path",  help="Path to the file containing the labels")
    parser.add_argument("-d",  "--dataset_path",      dest="dataset_path",      help="Path to the dataset folder (containing the input files)")
    parser.add_argument("-s",  "--save_path",         dest="save_path",         help="Path to the folder in which we want to save the results")

    parser.add_argument("-o",       "--overshoot",    dest="overshoot_df",    help="Overshoot parameter of the Deepfool algorithm", type=float)
    parser.add_argument("-maxiter", "--maxiter",      dest="max_iter_df",     help="Maximum number of iterations", type=int)
    parser.add_argument("-miniter", "--miniter",      dest="min_iter_df",     help="Minimum number of iterations", type=int)

    parser.add_argument("-target_class",  "--target_class",  dest="target_class",   help="The target class", type=int)

    args = parser.parse_args()


    #LOAD LABELS
    ####################
    labels_file_path = args.labels_file_path
    print("Path of the labels file: \t " + labels_file_path)
    labels = open(labels_file_path,'r').read().split('\n')


    #LABEL IDX
    ####################
    label_idx = args.label_idx
    print("Label idx: \t " + str(label_idx) + " (" + labels[label_idx] + ")")


    #TARGET MODEL PATH
    ####################
    model_path = args.model_path
    print("Path of the target model: \t " + model_path)


    #DATASET PATH
    ####################
    dataset_path = args.dataset_path
    

    #LOAD AUDIO FILENAMES
    ######################
    print("Loading audio filenames")
    input_files_path = args.input_files_path
    target_audios = np.load(input_files_path) #Load audios to execute

    #Select the audios in the range
    minidx = max(0, args.minidx)
    maxidx = min(len(target_audios), args.maxidx)
    if minidx!=-1 and maxidx!=-1:
        target_audios = target_audios[minidx:maxidx]
        print("-- Audios from index " + str(minidx) + " to " + str(maxidx) + " will be processed")
        #Sanity check:
        if len(target_audios)!=(maxidx-minidx): sys.exit("Wrong number of files copied")
    else:
        sys.exit("Pass valid minidx/maxidx parameters!")

    print("Audio filenames successfully loaded!")


    #GENERATE SETUP CLASS
    ######################
    print("Generating setup class...")
    setup = Attack(model_path=model_path, labels_file_path=labels_file_path)
    print("Class generated!")


    #LAUNCH DEEPFOOL
    ######################

    print("Launching deepfool")
    overshoot_df = args.overshoot_df
    max_iter_df  = args.max_iter_df
    min_iter_df  = args.min_iter_df
    target_class = args.target_class

    print("-- overshoot_df: " + str(overshoot_df))
    print("-- max_iter_df: "  + str(max_iter_df))
    print("-- min_iter_df: "  + str(min_iter_df))
    print("-- target_class: " + str(target_class))

    audio_cont = 0

    for audio_filename in target_audios:

        print(audio_filename)

        #Retrieve file extension
        filename_folder , filename_with_extension = re.match("(.+)/(.+)",audio_filename).groups()
        filename_no_extension = re.match("(.+).wav",filename_with_extension).groups()[0]
        filename_final_name = filename_folder + "_" + filename_no_extension

        full_audio_path = dataset_path + "/" + audio_filename
        fs, audio = wav.read(full_audio_path)

        scale_factor = 1/(1<<15)
        audio_scaled = audio*scale_factor
        audio_scaled = audio_scaled.reshape(1,16000)


        #Start watch
        t_start = time.process_time()  

        #Adversarial attack
        r_tot, adv_pert, loop_i, k_i,\
        pert_audio = setup.targeted_deepfool(audio=audio_scaled, 
                                            label=label_idx,
                                            target_class=target_class, 
                                            num_classes=12, 
                                            overshoot=overshoot_df, 
                                            max_iter=max_iter_df, 
                                            min_iter=min_iter_df,
                                            verbose=False)

        #Stop watch
        t_stop = time.process_time()  
        elapsed_time = t_stop-t_start

        #print("Deepfool finished!")



        #print("Saving results...")
        save_path = args.save_path
        #print(save_path)

        params_at_filename = "_t_" + str(target_class) + "_ov_" + str(overshoot_df)  + "_maxit_" + str(max_iter_df) 

        #Save the perturbation (considering  rounding constraints)
        filename = "adv_pert_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, adv_pert)

        #Save the total perturbation (raw, without rounding constraints)
        filename = "r_tot_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, r_tot)

        #Save the adversarial example
        filename = "pert_audio_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, pert_audio)

        #Save the final class
        filename = "k_i_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, k_i)

        #Save the number of iterations needed
        filename = "loop_i_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, loop_i)

        #Save the elapsed time in seconds
        filename = "secs_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, elapsed_time)

        print("Results successfully saved!")

        audio_cont = audio_cont + 1


    print("Final audio cont: %d"%audio_cont)

    print("Job done!")




