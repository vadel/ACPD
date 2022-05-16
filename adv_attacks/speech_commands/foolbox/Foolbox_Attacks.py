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

import foolbox
from foolbox.models import TensorFlowModel
from foolbox.criteria import Misclassification
from foolbox.criteria import TargetClass


#Helper function to reset the graph and set a random seed
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


def round_preprocessing(x):
    result = np.clip(x, -1.0, 1.0)
    result = np.array(np.clip(np.round(result*(1<<15)), -2**15, 2**15-1), dtype=np.int16)
    result = result/(1<<15)

    def grad(x):
        return x

    return result, grad



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
        print("\t- Tensors restored! Initialization sufccesfully completed!")

    #Compute logits from audio signal (shape 1x16000)
    def f(self, audio_sample):        
        output = self.sess.run(self.logits_tensor,  feed_dict={self.input_tensor: audio_sample})
        return output
    
    #Compute logits from audio signal (shape 1x16000) and ensures rounding constraints
    def f_round(self, audio_sample): 
        audio_sample = np.clip(audio_sample, -1.0, 1.0)
        audio_sample = np.array(np.clip(np.round(audio_sample*(1<<15)), -2**15, 2**15-1), dtype=np.int16)
        audio_sample = audio_sample/(1<<15)
        
        output = self.sess.run(self.logits_tensor,
                               feed_dict={self.input_tensor: audio_sample.reshape(1,16000)})
        return output
    
    
    #Set up foolbox
    def foolbox_model_setup(self, sel_attack, control_format):
        with self.sess.as_default():
            self.fb_model = TensorFlowModel(self.input_tensor, self.logits_tensor, bounds=(-1.0, 1.0))
        
        if control_format:
            print(">> Model setup: Enabled control format!")
            #Control here possible particularities of some attacks... e.g.:
            #if sel_attack in ["rand_pgd", "cw_l2", "fgsm_v2"]:
            if True:
                #In the current version, this will be applied to all the attacks:
                #self.fb_model.forward = self.f_round
                setup.fb_model._preprocessing = round_preprocessing  # Custom preprocessing operator
            else:
                sys.exit("Attack not supported")
    

    def foolbox_attack(self, sel_attack, input_sample, label, target_label=None, max_iters=30): 
        #Format tweaks for foolbox
        input_sample = input_sample.reshape(1,16000)
        label = np.array(label).reshape(1,)
        with self.sess.as_default():
            #Set the criterion
            if target_label is None: fb_criterion = Misclassification()
            else:                    fb_criterion = TargetClass(target_label)
                
            #Set the attack strategy
            if sel_attack=="bim_l2":
                attack = foolbox.attacks.L2BasicIterativeAttack(self.fb_model, fb_criterion,
                                                                foolbox.distances.MeanSquaredDistance)
            elif sel_attack=="pgd":
                attack = foolbox.attacks.PGD(self.fb_model, fb_criterion,
                                             foolbox.distances.Linfinity)
            elif sel_attack=="pgd_l2":
                attack = foolbox.attacks.PGD(self.fb_model, fb_criterion,
                                             foolbox.distances.MeanSquaredDistance)
            elif sel_attack=="rand_pgd":
                attack = foolbox.attacks.RandomPGD(self.fb_model, fb_criterion,
                                                   foolbox.distances.Linfinity)
            elif sel_attack=="df_fb_l2":
                attack = foolbox.attacks.DeepFoolAttack(self.fb_model, fb_criterion,
                                                        foolbox.distances.MeanSquaredDistance)
            elif sel_attack=="cw_l2":
                attack = foolbox.attacks.CarliniWagnerL2Attack(self.fb_model, fb_criterion,
                                                               foolbox.distances.MeanSquaredDistance)
            elif sel_attack in ["fgsm", "fgsm_v2"]:
                attack = foolbox.attacks.GradientSignAttack(self.fb_model, fb_criterion,
                                                            foolbox.distances.Linfinity)
            elif sel_attack=="jsma":
                attack = foolbox.attacks.SaliencyMapAttack(self.fb_model, fb_criterion,
                                                           foolbox.distances.MeanSquaredDistance)
            elif sel_attack=="noise_unif_l2":
                attack = foolbox.attacks.AdditiveUniformNoiseAttack(self.fb_model, fb_criterion,
                                                                    foolbox.distances.MeanSquaredDistance)
            elif sel_attack=="noise_gaus_l2":
                attack = foolbox.attacks.AdditiveGaussianNoiseAttack(self.fb_model, fb_criterion,
                                                                     foolbox.distances.MeanSquaredDistance)
            else:
                sys.exit("Attack not supported")
            
            #Launch the attacks
            if sel_attack in ["bim_l2", "pgd", "pgd_l2", "rand_pgd"]:
                pert_input = attack(input_sample, label, iterations=max_iters)
            elif sel_attack in ["df_fb_l2"]:
                pert_input = attack(input_sample, label, steps=max_iters, subsample=self.num_classes, p=2)
            elif sel_attack in ["cw_l2"]:
                #20 binary-search_steps used to match the default values of the previous attacks.
                pert_input = attack(input_sample, label, max_iterations=max_iters, binary_search_steps=20)
            elif sel_attack in ["jsma"]:
                pert_input = attack(input_sample, label, max_iter=max_iters, theta=0.001)
            elif sel_attack in ["fgsm"]:
                # (*) Read note below
                max_iters = np.linspace(0, 0.001, num=1000 + 1)[1:]
                pert_input = attack(input_sample, label, epsilons=max_iters)
            elif sel_attack in ["fgsm_v2"]:
                # (*) Read note below
                max_iters = np.linspace(0, 0.01, num=1000 + 1)[1:]
                pert_input = attack(input_sample, label, epsilons=max_iters)
            elif sel_attack in ["noise_unif_l2", "noise_gaus_l2"]:
                # (*) Read note below
                max_iters = np.linspace(0, 0.1, num=1000 + 1)[1:]
                pert_input = attack(input_sample, label, epsilons=max_iters)
            else:
                sys.exit("Supported attacks: [bim_l2, pgd, pgd_l2, rand_pgd, df_fb_l2, cw_l2...]")
            #(*) In these methods different epsilon values are tried. If epsilon is integer, N values will be
            #tried uniformly in the range (0,1). If epsilon is Iterable, those values passed are tried.
        
        
        #Final total (raw) perturbation
        r_tot   = pert_input.flatten() - input_sample.flatten()
        f_i_raw = np.array(self.f(pert_input)).flatten() #Logits of the perturbed input
        k_i_raw = int(np.argmax(f_i_raw)) #Class of the perturbed input
        
        #Ensure that the input satisfies format restrictions
        pert_input = np.clip(pert_input, -1.0, 1.0)
        pert_input = np.array(np.clip(np.round(pert_input*(1<<15)), -2**15, 2**15-1), dtype=np.int16)
        pert_input = pert_input/(1<<15)
        
        #Final perturbation considering restrictions
        adv_pert = pert_input.flatten() - input_sample.flatten() 
        f_i = np.array(self.f(pert_input)).flatten() #Logits of the perturbed input
        k_i = int(np.argmax(f_i)) #Class of the perturbed input

        return r_tot, adv_pert, k_i_raw, k_i, pert_input



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i",      "--input_filenames", dest="input_files_path",  help="Full path to the numpy file (.npy) containing input audio filenames")
    parser.add_argument("-l",      "--label_idx",       dest="label_idx",         help="Integer representing the label index", type=int)
    parser.add_argument("-m",      "--model_path",      dest="model_path",        help="Path to the frozen TF model (.pb)")
    parser.add_argument("-d",      "--dataset_path",    dest="dataset_path",      help="Path to the dataset folder (containing the audio files)")
    parser.add_argument("-lf",     "--labels_file_path",dest="labels_file_path",  help="Path to the file containing the labels (conv_actions_labels.txt)")
    
    parser.add_argument("-s",  "--save_path",           dest="save_path",         help="Path to the folder in which we want to save the results")

    parser.add_argument("-minidx", "--minidx",          dest="minidx",            help="Indicates the index of the first audio to process (from --input filenames)", type=int, default=-1)
    parser.add_argument("-maxidx", "--maxidx",          dest="maxidx",            help="Indicates the index of the last  audio to process (from --input filenames)", type=int, default=-1)

    parser.add_argument("-maxiter", "--maxiter",        dest="max_iter",   help="Maximum number of iterations", type=int)
    parser.add_argument("-miniter", "--miniter",        dest="min_iter",   help="Minimum number of iterations", type=int)
    
    parser.add_argument("-sel_attack",   "--sel_attack",   dest="sel_attack",   help="Attack name: [bim_l2, pgd]")
    parser.add_argument("-target_class", "--target_class", dest="target_class", help="The target class we want (None for untargeted)", type=int, default=None)

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


    #LAUNCH THE ATTACK
    ######################

    print("Launching attack")
    max_iter     = args.max_iter
    min_iter     = args.min_iter
    sel_attack   = args.sel_attack
    target_class = args.target_class

    print("-- max_iter: "  + str(max_iter))
    print("-- min_iter: "  + str(min_iter))
    print("-- sel_attack:   " + str(sel_attack))
    print("-- target_class: " + str(target_class))


    #GENERATE SETUP FOR FOOLBOX
    ############################
    setup.foolbox_model_setup(sel_attack=sel_attack, control_format=True)
    print("Foolbox model setup finished!")

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
        r_tot, adv_pert, k_i_raw, k_i,\
        pert_audio = setup.foolbox_attack(sel_attack=sel_attack,
                                          input_sample=audio_scaled,
                                          label=label_idx,
                                          target_label=target_class,
                                          max_iters=max_iter)

        #Stop watch
        t_stop = time.process_time()  
        elapsed_time = t_stop-t_start

        print("Attack finished!")



        #print("Saving results...")
        save_path = args.save_path
        #print(save_path)
        
        if target_class is None:
            params_at_filename = "_maxit_" + str(max_iter) 
        else:
            params_at_filename = "_t_" + str(target_class) + "_maxit_" + str(max_iter) 

        #Save the perturbation (considering  rounding constraints)
        filename = "adv_pert_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, adv_pert)

        #Save the total perturbation (raw, without rounding constraints)
        filename = "r_tot_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, r_tot)

        ##Save the adversarial example
        #filename = "pert_audio_" + filename_final_name + params_at_filename + ".npy"
        #np.save(save_path + filename, pert_audio)

        #Save the final class (considering  rounding constraints)
        filename = "k_i_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, k_i)

        #Save the final class (raw, without rounding constraints)
        filename = "k_i_raw_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, k_i_raw)

        #Save the elapsed time in seconds
        filename = "secs_" + filename_final_name + params_at_filename + ".npy"
        np.save(save_path + filename, elapsed_time)

        #print("Results successfully saved!")

        audio_cont = audio_cont + 1


    print("Final audio cont: %d"%audio_cont)

    print("Job done!")




