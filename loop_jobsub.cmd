#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")
wd_dir="/tigress/abeukers/wd/epm"

CELLSIZE=${1}

## now loop through the above array
for NUMBACK in {2..6}; do 
	for NUMSTIM in {10..15}; do 
		sbatch ${wd_dir}/gpu_jobsub.cmd "${NUMBACK}" "${NUMSTIM}" 
	done
done




