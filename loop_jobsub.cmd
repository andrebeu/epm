#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")
wd_dir="/tigress/abeukers/wd/epm"


declare -a stsize_arr=(30 40 50)
declare -a ogtoken_arr=(2 3 4)
declare -a pmtrial_arr=(2 3 4 5)

for seed in {10..15}; do 
	for stsize in "${stsize_arr[@]}"; do 
		for ogtoken in "${ogtoken_arr[@]}"; do 
			for pmtrials in "${pmtrial_arr[@]}"; do 
				sbatch ${wd_dir}/gpu_jobsub.cmd "${stsize}" "2" "${ogtoken}" "${pmtrials}" "${seed}"
				sbatch ${wd_dir}/gpu_jobsub.cmd "${stsize}" "3" "${ogtoken}" "${pmtrials}" "${seed}"
				sbatch ${wd_dir}/gpu_jobsub.cmd "${stsize}" "4" "${ogtoken}" "${pmtrials}" "${seed}"
			done
		done
	done
done
