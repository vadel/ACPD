#!/bin/bash
#generate main folders for the optimization results
task="speech_commands"
for attack in deepfool cw_l2 fgsm_v2 rand_pgd
do
	echo "Processing attack: $attack"
	for method in mab mfrb m1 m2 m3 m4 m4_1 m4_2
	do
		echo "\tMethod type: $method"
		for pyset in 1 2
		do
			#echo "\t\tP(Y)  set: $pyset"
			for ntrain in 1 10 50 100 500
			do
				echo "\t\t\t generating  $task/$attack/results_$method/p_y_obj_set_$pyset/N_train_$ntrain"
				mkdir -p "$task/$attack/results_$method/p_y_obj_set_$pyset/N_train_$ntrain"
			done
		done
	done
done
