#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")
wd_dir="/tigress/abeukers/wd/epm"

CELLSIZE=${1}

## now loop through the above array
for i in {1..5}; do 
	sbatch ${wd_dir}/gpu_jobsub.cmd "${CELLSIZE}" 
done




