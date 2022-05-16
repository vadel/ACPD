#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import re
import pulp
import pyomo
from scipy.stats import entropy
import scipy as sp
import collections


# Maximum Approximation Baseline (MAB) described in the paper, in which the target classes
# will be sampled according to the probabilities defined by the target probability distribution.
# In order to use a similar implementation to that used for the main methods introduced in the paper,
# we will create a transition matrix in which all the rows have the target distribution.
def generate_T_MAB(p_y_obj, num_classes):
    T_norm = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        T_norm[i, :] = np.copy(p_y_obj)  # Assign the objective distribution p_y_obj to all the rows

    # To mimic the behaviour of Pulp, we also return solver_status with "Optimal" as value
    # ("Optimal" is achieved when the linear program is feasible and an optimal solution is found)
    solver_status = "Optimal"

    return T_norm, solver_status


# Maximum Fooling Rate Baseline (MFRB) described in the paper.
# In order to use a similar implementation to that used for the main methods introduced in the paper,
# we will create a transition matrix in which all the rows have the target distribution, although a zero will be set
# in the diagonal in order to maximize the fooling rate (each row will be "normalized" accordingly to ensure that
# each row has a probability distribution of the classes).
def generate_T_MFRB(p_y_obj, num_classes):
    '''
    @param p_y_obj:  numpy array containing the target probability distribution for the classes
    @param num_classes: number of classes
    '''
    T_norm = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        T_norm[i, :] = np.copy(p_y_obj)  # Assign the target distribution p_y_obj to all the rows
        T_norm[i, i] = 0.0               # Assign zero to the "ground-truth" class (to maximize fooling ratio)
        T_norm[i, :] = T_norm[i, :] / np.sum(T_norm[i, :])  # 'Normalize' the row to be a valid probability distribution

    # To mimic the behaviour of Pulp, we also return solver_status with "Optimal" as value
    # ("Optimal" is achieved when the linear program is feasible and an optimal solution is found)
    solver_status = "Optimal"

    return T_norm, solver_status



#Method 1 (Agnostic Method)
def generate_T_agnostic(p_y_init, R, p_y_obj, num_classes):
    '''
    @param p_y_init: numpy array containing the initial probability distribution for the classes
    @param p_y_obj:  numpy array containing the target probability distribution for the classes
    @param num_classes: number of classes
    '''

    # We define the problem as of minimization
    exercise1 = pulp.LpProblem("Exercise1", pulp.LpMinimize)

    #The main decision variable is T, the transition matrix (square of order N_class)
    T = pulp.LpVariable.dicts("T",
                             ((i, j) for i in range(num_classes) for j in range(num_classes)),
                             lowBound=0.0, upBound=1.0,
                             cat='Continuous')

    #Auxiliary variable (lower bounds for each value in T)
    LB = pulp.LpVariable.dicts("LB",
                                ((i,j) for i in range(num_classes) for j in range(num_classes)),
                                lowBound=0.0, upBound=0.01,
                                cat='Continuous')

    ### Objective function ###
    ##########################
    #Criterion 1: minimizing the diagonal of the transition matrix
    term1 = sum([T[(i,i)] for i in range(num_classes)])
    #Criterion 2: maximizing the number of nonzero probabilities.
    term2 = sum([LB[(i,j)] for i in range(num_classes) for j in range(num_classes) if i!=j] )
    exercise1 += term1 - term2, "z"

    ### Constraints ###
    ###################
    #Ensure that the sum of each row of T equals 1 (T is a transition matrix)
    for i in range(num_classes):
        exercise1 += sum([T[(i,j)] for j in range(num_classes)]) == 1

    #Ensure that P_init(Y)*T = P_obj(Y)
    for j in range(num_classes):
        exercise1 += sum([p_y_init[i]*T[(i,j)] for i in range(num_classes)]) == p_y_obj[j]

    #Set the lower bounds for T (except in the main diagonal)
    for i in range(num_classes):
        for j in range(num_classes):
            if i!=j:
                exercise1 += T[(i,j)] >= LB[(i,j)]
                
    exercise1.solve() # The solver is invoked

    solver_status = pulp.LpStatus[exercise1.status]
    #Retrieve the values from T and store them in a numpy array
    T_norm = np.zeros((num_classes,num_classes))
    for variable in exercise1.variables():
        if variable.name[0]!="T":
            continue
        val1 = int(re.search(r'\d+',variable.name).group(0)) #row-index
        val2 = int(re.search(r'.+\d,_(\d+)',variable.name).group(1)) #col-index
        T_norm[val1,val2] = variable.varValue #save the value
    
    return T_norm, solver_status


#Method 2 - Upper-bound Method (UBM)
def generate_T_method_2(p_y_init, samples_per_class, R, p_y_obj, num_classes):
    '''
    @param p_y_init: numpy array containing the initial probability distribution for the classes
    @param samples_per_class: numpy array containing the number of samples per class considered to construct R
    @param R: R matrix defined in Equation 6 (numpy matrix)
    @param p_y_obj:  numpy array containing the target probability distribution for the classes
    @param num_classes: number of classes
    '''

    # Normalize the matrix R by the number of samples per class (see Equation 7)
    R = np.array([R[i,:]/float(samples_per_class[i]) for i in range(num_classes)])

    #We define the problem as of maximization
    exercise1 = pulp.LpProblem("Exercise1", pulp.LpMinimize)

    #The main decision variable is T, the transition matrix (square of order N_class)
    T = pulp.LpVariable.dicts("T",
                              ((i,j) for i in range(num_classes) for j in range(num_classes)),
                              lowBound=0.0, upBound=1.0,
                              cat='Continuous')

    #Auxiliary variable
    LB = pulp.LpVariable.dicts("LB",
                               ((i,j) for i in range(num_classes) for j in range(num_classes)),
                               lowBound=0.0, upBound=0.01,
                               cat='Continuous')

    #Auxiliary variable
    UB = pulp.LpVariable.dicts("UB",
                               ((i,j) for i in range(num_classes) for j in range(num_classes)),
                               lowBound=0.0, upBound=1.0,
                               cat='Continuous')

    ### Objective function ###
    ##########################
    #Criterion 1: minimizing the diagonal of the transition matrix
    term1 = sum([T[(i,i)] for i in range(num_classes)])
    #Criterion 2: maximizing the number of nonzero probabilities.
    term2 = sum([LB[(i,j)] for i in range(num_classes) for j in range(num_classes) if i!=j] )
    #Criterion 3: minimizing the upper bound (width parameter).
    term3 = sum([UB[(i,j)] for i in range(num_classes) for j in range(num_classes)] )
    exercise1 += term1 + 10.0*term3 - term2, "z"

    ### Constraints ###
    ###################
    #Ensure that the sum of each row of T equals 1 (T is a transition matrix)
    for i in range(num_classes):
        exercise1 += sum([T[(i,j)] for j in range(num_classes)]) == 1

    #Ensure that P_init(Y)*T=P_obj(Y)
    for j in range(num_classes):
        exercise1 += sum([p_y_init[i]*T[(i,j)] for i in range(num_classes)]) == p_y_obj[j]

    #Ensure that we do not assign more cases than the ones specified in R (relaxed by UB)
    for i in range(num_classes):
        for j in range(num_classes):
            exercise1 += T[(i,j)] <= (R[(i,j)] + UB[(i,j)])

    #Lower bounds
    for i in range(num_classes):
        for j in range(num_classes):
            if i!=j:
                exercise1 += T[(i,j)] >= LB[(i,j)]

    exercise1.solve() # The solver is invoked

    solver_status = pulp.LpStatus[exercise1.status]

    #Retrieve the values from T and store them in a numpy array
    T_norm = np.zeros((num_classes,num_classes))
    for variable in exercise1.variables():
        if variable.name[0]!="T":
            continue
        val1 = int(re.search(r'\d+',variable.name).group(0))
        val2 = int(re.search(r'.+\d,_(\d+)',variable.name).group(1))
        T_norm[val1,val2] = variable.varValue

    return T_norm, solver_status



#Method 3 - Element-wise Transformation Method (EWTM)
def generate_T_method_3(p_y_init, samples_per_class, R, p_y_obj, num_classes):
    '''
    @param p_y_init: numpy array containing the initial probability distribution for the classes
    @param samples_per_class: numpy array containing the number of samples per class considered to construct R
    @param R: R matrix defined in Equation 6 (numpy matrix)
    @param p_y_obj:  numpy array containing the target probability distribution for the classes
    @param num_classes: number of classes
    '''

    #"Normalize" the matrix R (as described in Section 3.4)
    R = np.array([R[i,:]/np.sum(R[i,:]) for i in range(num_classes)])
    
    # We define the problem as of maximization
    exercise1 = pulp.LpProblem("Exercise1", pulp.LpMinimize)

    #The main decision variable is Q, a 'transformation' matrix (square of order N_class)
    Q = pulp.LpVariable.dicts("Q",
                              ((i, j) for i in range(num_classes) for j in range(num_classes)),
                              lowBound=0.0,
                              cat='Continuous')

    #Auxiliary variable
    UB = pulp.LpVariable('UB', lowBound=0, cat='Continuous')

    ### Objective function ###
    ##########################
    term1 = sum([R[(i,i)]*Q[(i,i)] for i in range(num_classes)])
    term2 = UB
    exercise1 += term1 + term2, "z"

    ### Constraints ###
    ###################
    #Ensure that the sum of each row of T=R*Q equals 1 (T=R*Q is a transition matrix)
    for i in range(num_classes):
        exercise1 += sum([R[(i,j)]*Q[(i,j)] for j in range(num_classes)]) == 1

    for i in range(num_classes):
        for j in range(num_classes):
            exercise1 += R[(i,j)]*Q[(i,j)] <= 1 #Ensure that Tij=Rij*Qij <= 1  (transition matrix)
            exercise1 += R[(i,j)]*Q[(i,j)] >= 0 #Ensure that Tij=Rij*Qij => 0  (transition matrix)
            exercise1 += Q[(i,j)] <= UB #Try to restrict the values of Q...
    
    #Ensure that P_init(Y)*T=P_obj(Y)
    for j in range(num_classes):
        exercise1 += sum([p_y_init[i]*(R[(i,j)]*Q[(i,j)]) for i in range(num_classes)]) == p_y_obj[j]

    ### The solver is invoked ###
    #############################
    exercise1.solve()

    solver_status = pulp.LpStatus[exercise1.status]
    
    # The optimal values for each of the variables are displayed
    Q_opt = np.zeros((num_classes,num_classes))
    for variable in exercise1.variables():
        if variable.name[0]!="Q":
            continue
        val1 = int(re.search(r'\d+',variable.name).group(0))
        val2 = int(re.search(r'.+\d,_(\d+)',variable.name).group(1))
        Q_opt[val1,val2] = variable.varValue

    T_norm = R*Q_opt

    return T_norm, solver_status


#Auxiliary function for Method 4 (Chain-rule - CR)
def compute_subset_probs(reachability, ground_truth, num_classes,
                        all_nonzero=False, ignore_non_foolable=False):
    '''
    @param reachability: numpy matrix containing, for a set of inputs (row-wise), which classes (column-wise) can be reached from them
                         (=1 if can be reached, =0 otherwise).
    @param ground_truth: numpy array ground-truth classes for the inputs in 'reachability'.
    @param num_classes: number of classes (integer).
    @param all_nonzero: if set to True, Applies the Laplace correlation to the probabilities P(S={y_i}|y_i).
    @param ignore_non_foolable: if set to True, the probabilities P(S={y_i}|y_i) are set to zero.
    '''

    #Compute the probabilities
    subset_probs = np.zeros((num_classes,1<<num_classes), dtype=float)
    for i in range(reachability.shape[0]):
        row = np.copy(reachability[i,:])
        label_idx = int(ground_truth[i])
        pos = int( np.sum( [row[l]*(1<<l) for l in reversed(range(num_classes))] ) )
        if label_idx not in [l for l in range(num_classes) if (pos & (1<<l)) != 0]:
            sys.exit("Error in the computation of the subset probabilities")
        subset_probs[label_idx, pos] += 1
    
    if all_nonzero:
        #Apply the Laplace correction
        for i in range(num_classes):
            for i_s in range(1<<num_classes):
                #Skip those subsets in which 'y_i' is not an element
                if (i_s & (1<<i))!=0:
                    subset_probs[i,i_s]+=1

    if ignore_non_foolable:
        #Set to zero the probs P(S={y_i}|y_i).
        for i in range(num_classes):
            subset_probs[i,(1<<i)] = 0

    #Normalize the distributions P(S|y_i) \forall i to ensure they are prob. distrs.
    for i in range(num_classes):
        subset_probs[i,:] = subset_probs[i,:]/np.sum(subset_probs[i,:])
    
    return subset_probs


#Method 4 - Chain-rule Method (CRM)
def generate_T_method_4(p_y_init, p_y_obj, subset_probs, num_classes):
    '''
    @param p_y_init: numpy array containing the initial probability distribution for the classes
    @param p_y_obj:  numpy array containing the target probability distribution for the classes
    @param subset_probs: probability of each possible set of reachable classes, for each class
    @param num_classes: number of classes
    '''
    
    # We define the problem as of maximization
    exercise1 = pulp.LpProblem("Exercise1", pulp.LpMinimize)

    # Auxiliary variable to help creating the decision variables for the problem
    # This variable is a map to know which probabilities P(y_j,S,y_i) are valid
    # (i.e., a prob is valid only if both y_i and y_j are elements of S
    valid_V = np.zeros((num_classes, 1<<num_classes, num_classes), dtype=bool)
    for i in range(num_classes):
        for i_s in range(1<<num_classes):
            labels_in_set = [l for l in range(num_classes) if (i_s & (1<<l))!=0]
            if i in labels_in_set:
                for j in labels_in_set:
                    valid_V[i, i_s, j] = True
                    
    # Decision variables
    V = pulp.LpVariable.dicts("V",
                             ((i, i_s, j) for i in range(num_classes) for i_s in range(1<<num_classes) for j in range(num_classes) if valid_V[i,i_s,j]),
                             lowBound=0.0, upBound=1.0,
                             cat='Continuous')
    
    #Auxiliary function to map V_is to T_ij
    def get_tij(subset_probs, V, i, j):
        out = 0
        #For each label, process each possible subset
        for i_s in range(1<<num_classes):
            labels_in_set = [l for l in range(num_classes) if (i_s & (1<<l)) != 0]
            if i in labels_in_set and j in labels_in_set:
                out += subset_probs[i,i_s]*V[i,i_s,j]
        return out
    
    #####################
    # Objective function
    #####################
    term1 = sum([get_tij(subset_probs, V, i, i) for i in range(num_classes)])
    exercise1 += term1, "z"

    ##############
    # Constraints
    ##############
    #Ensure that the sum of the probs of each subset sums 1
    for i_l in range(num_classes):
        for i_s in range(1<<num_classes):
            labels_in_set = [l for l in range(num_classes) if (i_s & (1<<l)) != 0]
            if i_l in labels_in_set:
                tmp = sum( [1 for j in range(num_classes) if valid_V[i_l,i_s,j]] )
                if tmp==0:
                    sys.exit("Trying to fix restrictions to a set of empty reachable classes!")
                exercise1 += sum([V[(i_l,i_s,j)] for j in range(num_classes) if valid_V[i_l,i_s,j]]) == 1

    #Ensure that we achieve the target distribution --> P_init(Y)*T=P_obj(Y)
    for j in range(num_classes):
        exercise1 += sum([p_y_init[i]*get_tij(subset_probs,V,i,j) for i in range(num_classes)]) == p_y_obj[j]

    # The solver is invoked
    exercise1.solve()

    solver_status = pulp.LpStatus[exercise1.status]

    #Retrieve the optimal values
    V_opt = np.zeros((num_classes,1<<num_classes,num_classes))
    for variable in exercise1.variables():
        if variable.name[0]!="V":
            continue
        val1 = int(re.search(r'\d+',variable.name).group(0))
        val2 = int(re.search(r'.+\(\d+,_(\d+)',variable.name).group(1))
        val3 = int(re.search(r'.+\d+,_(\d+),_(\d+)',variable.name).group(2))
        V_opt[val1,val2,val3] = variable.varValue

    #Some changes in the names and formats...
    T_norm = np.zeros((num_classes,num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            T_norm[i,j] = get_tij(subset_probs, V_opt, i, j)
 
    #Start checking:
    return T_norm, V_opt, solver_status
