#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")
wd_dir="/tigress/abeukers/wd/epm"


## now loop through the above array
for NUMBACK in {1..5}; do 
	sbatch ${wd_dir}/gpu_jobsub.cmd "${NUMBACK}" 
done
