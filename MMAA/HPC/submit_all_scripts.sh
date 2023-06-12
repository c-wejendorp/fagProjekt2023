#!/bin/sh
bsub < MMAA/HPC/submit_scripts/submit_MMAA_GPU_split-0_arg_num-0.sh
bsub < MMAA/HPC/submit_scripts/submit_MMAA_GPU_split-0_arg_num-1.sh
bsub < MMAA/HPC/submit_scripts/submit_MMAA_GPU_split-0_arg_num-2.sh
bsub < MMAA/HPC/submit_scripts/submit_MMAA_GPU_split-0_arg_num-3.sh

bsub < MMAA/HPC/submit_scripts/submit_MMAA_GPU_split-1_arg_num-0.sh
bsub < MMAA/HPC/submit_scripts/submit_MMAA_GPU_split-1_arg_num-1.sh
bsub < MMAA/HPC/submit_scripts/submit_MMAA_GPU_split-1_arg_num-2.sh
bsub < MMAA/HPC/submit_scripts/submit_MMAA_GPU_split-1_arg_num-3.sh



