#!/bin/sh

### select queue 
#BSUB -q hpc

### name of job, output file and err
#BSUB -J test_loss
#BSUB -o test_loss_%J.out
#BSUB -e test_loss_%J.err


### number of cores
#BSUB -n 1


### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need rather large amount of of Memory 
### as we need to load the X matrix and perform calculations on it

#BSUB -R "rusage[mem=24GB]"

### wall time limit - the maximum time the job will run. Currently 2 hours 30 min. 
### one modal comb takes longer than 30 min

### this might need to be adjusted 

#BSUB -W 02:30

##BSUB -u s204090@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 


# end of BSUB options


# load the correct  scipy module and python

module load scipy/1.10.1-python-3.11.3
module load cuda/11.8


# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source MMAA/HPC_env/bin/activate

python MMAA/calc_test_loss.py
