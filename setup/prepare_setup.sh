#!/bin/bash
#Get to the project root
cd ../
rootdir="$(pwd)"

echo "Creating required directories..."
mkdir datasets/
#Create the folders for the attacks in the speech command recognition problem
for attack in deepfool cw_l2 fgsm_v2 rand_pgd
do
	for att_type in targ untarg
	do
		for label in _silence_ _unknown_ yes no up down left right on off stop go
		do
			mkdir -p "adv_attacks/speech_commands/$attack/results_$att_type/$label"
		done
	done
done
#Create the analysis folders (speech command dataset)
mkdir adv_attacks/speech_commands/analysis/targeted/
mkdir adv_attacks/speech_commands/analysis/untargeted/
#Create the folders for the attacks in the tweet emotion problem
mkdir -p adv_attacks/text/genetic/results_targ
mkdir -p adv_attacks/text/genetic/results_untarg
#Create the analysis folders (text emotion dataset)
for mid in 0 1 2 3
do
  mkdir -p "adv_attacks/text/analysis/targeted/emotion/m$mid"
  mkdir -p "adv_attacks/text/analysis/untargeted/emotion/m$mid"
done
mkdir -p adv_attacks/text/analysis/predictions/

# Create folders to save the results of the proposed methods (in ./optimization/).
for attack in deepfool cw_l2 fgsm_v2 rand_pgd
do
	for method in mab mfrb m1 m2 m3 m4 m4_1 m4_2
	do
		for setidx in 1 2
		do
		  for ntrain in 1 10 50 100 500
		  do
			  mkdir -p "optimization/speech_commands/$attack/results_$method/p_y_obj_set_$setidx/N_train_$ntrain/"
			done
		done
	done
done

# Create folders for the label-shift experiments (in ./optimization/).
mkdir -p optimization/label_shift/emotion/genetic

echo "Directories created!"


#MODELS
echo "Downloading text classifier..."
cd models/
git clone https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion
cd $rootdir
echo "Done!"


#DATASETS
echo "Downloading speech command dataset..."
cd datasets/
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
echo "Unzipping..."
mkdir speech_commands_v0.02
tar -xf speech_commands_v0.02.tar.gz -C speech_commands_v0.02/
cd speech_commands_v0.02
wget https://www.dropbox.com/s/6jh1b7tymzno3zk/conv_actions_labels.txt
wget https://www.dropbox.com/s/vsy381la1bptv67/silence.tar.gz
tar -xf silence.tar.gz
cd $rootdir
echo "Done!"

echo "Downloading emotion dataset and resources..."
cd setup/
python3 Setup_Experiments_Text.py
cd $rootdir
echo "Done!"

echo "Setup successfully configured!"

