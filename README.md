# ACPD
This repository contains the code corresponding to the paper *Extending Adversarial Attacks to Produce Adversarial Class Probability Distributions*, by
Jon Vadillo, Roberto Santana & Jose A. Lozano (Under Review).

## Setup instructions
The following instructions describe how to download and create the required resources:

### 1. Install the dependencies
Install the required packages, which can be found in:

`setup/requirements_audio.txt` (Python 3.6.8)

In some steps of the project (i.e., in order to generate the adversarial examples in the text domain), the following requirements will be needed instead:

`setup/requirements_text.txt` (Python 3.8)

Installing two virtual environments, one for each set of requirementes, is strongly recommended (e.g., *venv_speech* and *venv_text*, respectively). 

### 2. Activate the virtual machine corresponding to the text requirements

    source venv_text/bin/activate
    
### 3. Execute the setup script

    cd setup
    sh prepare_setup.sh
    
  
## Main experimental pipeline
The code is organized in two different parts: the generation of adversarial examples and the generation of adversarial class probability distributions. In the first part, the adversarial examples are precomputed, for the sake of efficiency. In the second part, the effectiveness of the attack methods proposed (which employ different strategies in order to select the target class for every incoming input) is evaluated. The implementation of our methods can be found in `optimization/acpd_methods.py`.

In order to ease the execution of our programs, all the main executable files are acommpanied with a "launcher" (a python script), in which the required parameters and options can be specified (the default configuration is set in each of them).

### 1. Generation of adversarial examples in the Speech Command Dataset
The code required in this part can be found in the `./adv_attacks` directory, which contains two subfolders, one for each of the two problems considered in our paper: 
- speech command classification (`speech_commands/`)
- tweet emotion recognition (`text/`)

The main experiments of our paper are focused on the speech command classification problem. For the sake of efficiency, the adversarial examples will be precoumputed for a set of N=1000 samples per class, so that, afterwards, the methods proposed to generate the adversarial class probability distribution can be evaluated without the need of repeating the same computation several times.

**1.1.** Load the virtual machine corresponding to this problem.

    source venv_speech/bin/activate
    
**1.2.** Execute the following script:

    cd adv_attacks/speech_commands/deepfool/
    python3 targeted_launcher.py

which will perform the following tasks, for each pair of source and target classes:
+ Load a set of N input samples belonging to the source class.
+ Generate, for each input, an adversarial example targeting each of the classes in the problem, using the DeepFool algorithm.
+ Save the results in `adv_attacks/speech_commands/deepfool/results_targ/<source_label>/`. For each input, the following information will be stored: the adversarial perturbation, the number of iterations required, the elapsed time and the predicted "adversarial" class (to check if the attack successfully managed to produce the target class).
    
Notice that the adversarial examples will be generated using the [DeepFool](https://doi.org/10.1109/CVPR.2016.282) attack method. In order to use attacks other than DeepFool, use the script `targeted_launcher.py` located in the directory `adv_attacks/speech_commands/foolbox/` instead, which makes use of the [Foolboox](https://foolbox.readthedocs.io/en/v2.0.0/) package. The desired attack can be selected using the `sel_attack` variable, and the results will be saved in `adv_attacks/speech_commands/<sel_attack>/results_targ/<source_label>/` (more details can be found in the script).


**1.3.** Once all the results corresponding to an adversarial attack have been obtained, they can be "packed" using the following script:

    cd adv_attacks/speech_commands/analysis/
    python3 pack_results_targeted.py

which will save different numpy matrices in the `adv_attacks/speech_commands/analysis/targeted/` directory. The dimension of the matrices is `N*k`, being `N` the number of samples considered in the experiment and `k` the number of classes in the problem. The cell `(i,j)` of each matrix provides different information about the targeted adversarial attack generated for the input i&#8714;{0,...,N-1} targeting the class j&#8714;{0,...,k-1}:

- `reachability.npy`:  whether an adversarial example targetting the class `j` was successfully found `(=1)` or not `(=0)`.
- `num_steps_vec.npy`: the number of steps/iterations required to generate the attack.
- `l2_norm_vec.npy`:   the L2 norm of the adversarial perturbation (a file `linf_norm_vec.npy` will be also stored to handle those attacks which are based on the L<sub>&infin;</sub> norm).

These matrices will be used in the second part of the project, in order to evaluate the effectiveness of our methods in different "scenarios" involving different *"attack budgets"* (e.g., considering different thresholds regarding the amount of perturbation and regarding the number of steps of the attack). Thus, the aforementioned matrices will be used to know which targeted attacks are feasible given a specific attack budget. 




### 2. Adversarial Class Probability Distributions: Evaluating the methods introduced in the paper
The code corresponding to this part is located in the *optimization/* directory. The implementation of our methods can be found in the file `acpd_methods.py`.

**2.1.** Execute the following file in the directory *optimization/*:
    
    cd optimization/
    python3 launcher_acpd_experiments.py

by specifying:
+ `sel_attack`: the selected adversarial attack method.
+ `p_y_obj_set_idx`: the set of target distributions (either a single uniform distribution or a set of 100 random Dirichlet distributions).
+ `sel_method`: the method used to optimize the transition matrix that will guide the adversarial attack in order to produce the target distributions: *AM, UBM, EWTM, CRM, MAB* or *MFRB* (more details in the paper and in the script).
+ `max_iter_thr`: the maximum number of iterations allowed for each adversarial attack.
+ `max_dist_thr`: the maximum distortion amount allowed for each adversarial perturbation (i.e., &epsilon;).

This launcher will execute the *ACPD_experiments_complete.py* script, in which the whole attack process is implemented, as described below:
+ *2.1.1.* The results described in the step 1.3 will be loaded and filtered based on the thresholds specified as parameters. The filtering process ensures that those adversarial examples which surpass any of the thresholds are not considered (i.e., that the corresponding target class cannot be reached from the corresponding input without exceeding the attack budget).
+ *2.1.2.* The set of samples for which the adversarial examples were generated will be randomly split into two disjoint subsets: One of them will be used to generate the transition matrices (if required) and the other one to evaluate the effectiveness of those transition matrices on different samples. The total number of inputs and the number of inputs to be used for the generation of the matrices are defined in the launcher file, in the `N_per_class` and the `N_per_class_train` variables, respectively.
+ *2.1.3.* For each target distribution, the following process will be carried out:
  + The transition matrix will be generated, using the strategy corresponding to the attack method selected.
  + The attack process will be carried out, in which the transition matrix will be used to decide the target class for each new input (taking into account which target classes can be reached without exceeding the attack budget).
  + Once all the inputs have been processed, the empirical distribution of the classes will be computed (i.e., with what proportions the model has predicted each of the classes).
  + The similarity between the empirical and the target distributions will be measured using the similarity metrics described in the paper. 
  + The following results will be saved for each target distribution (the output directory is specified by the *save_path* variable, in the launcher script):
    + Whether a feasible transition matrix was found by the method selected.
    + The similarity between the target distribution and the one empirically generated by the attack process.
    + The fooling rate of the attack (i.e., the proportion of samples for which an incorrect class was selected as the target).
    + The original classes of the inputs considered in the evaluation and the ones produced by the attack process (i.e., those target classes randomly selected by the transition matrix).

This process will be repeated by swapping the two sets used in the step 2.1.2 (i.e., a 2-fold cross-validation will be carried out). Furthermore, the 2-fold cross-validation process can be repeated N<sub>rep</sub> times, using a different random partition of the input samples each time (see the `N_multistarts` parameter in the launcher file). By default, N<sub>rep</sub> = 50.


**2.2.** Visualize the results

In order to compute and visualize the average results of the experiments, use the following Jupyter Notebook: 

`visualization/Visualize_General_Comparison.ipynb`

In order to visualize the source, target and generated distributions for one particular case, use the following Jupyter Notebook:

`visualization/Visualize_Individual_Results.ipynb`


## Illustrative use case: Counteracting label-shift detectors

### 1. Generation of adversarial examples in the Tweet Emotion Dataset

**1.1.** Load the virtual machine corresponding to this problem:

    source venv_text/bin/activate

**1.2.** Precompute, for a set of randomly sampled inputs, an adversarial example targeting each possible class:

    cd adv_attacks/text/openattack/
    python3 launcher_attacks.py

The default attack strategy is the genetic-algorithm-based method proposed by [Alzantot et al., 2018](http://dx.doi.org/10.18653/v1/D18-1316). 
The [OpenAttack](https://openattack.readthedocs.io/en/latest/) package is used to generate the attacks. The results will be saved in `adv_attacks/text/genetic/results_targ/`.

**1.3.** Once all the attacks have been computed, all the results can be "packed" using the following script:

    cd adv_attacks/text/openattack/
    python3 launcher_pack_tar.py
    
which will save different numpy matrices in the directory `adv_attacks/text/analysis/targeted/emotion/m1/`. The dimension of the matrices is `N*k`, being `N` the number of samples considered in the experiment and `k` the number of classes in the problem. The cell `(i,j)` of each matrix provides different information about the targeted adversarial attack generated for the input i&#8714;{0,...,N-1} targeting the class j&#8714;{0,...,k-1}:

- `reachability.npy`:  Whether an adversarial example targetting the class `j` was successfully found `(=1)` or not `(=0)`.
- `num_steps_vec.npy`: The number of model queries required to generate the attack.
- `dist_vec.npy`:   The normalized Levenshtein Edit Distance between the original text and the adversarial text (more details in the paper).


### 2. Producing label-shifts without triggering detection methods

The experiments can be executed using the Jupyter Notebook (using the `venv_speech` virtual environment):

`optimization/label_shift/Label_Shift_Evaluation_TweetClassification.ipynb`. 

The main steps of the experiment are as follows:
+ The source probability distribution is selected (the possible configurations can be found in the notebook or in the paper).
+ The results corresponding to the adversarial examples computed in the steps 1.2 and 1.3 are loaded.
+ The reach of the attacks is filtered taking into account the attack budget, and two subsets of inputs are sampled, one to generate the transition matrices and the other one to evaluate the effectiveness of our attack method.
+ A set of 1000 target probability distributions is randomly sampled.
+ For each target distribution:
    + The transition matrix is generated, using, by default, the Element-wise Transformation Method (EWTM).
    + Our attack process is applied.
    + For cumulative batches of inputs, the label-shift detection approach is used to evaluate whether the empirical distribution is significantly different from the source distribution. The similarity between the empirical distribution and the target distribution is also measured.
    + The resuts are saved in the directory `optimization/label_shift/emotion/genetic/`

The average result (for the different metrics considered) can be computed using the Jupyter Notebook 

`optimization/label_shift/Label_Shift_EvaluationSummary.ipynb`.
