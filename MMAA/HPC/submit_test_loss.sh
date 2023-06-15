#!/bin/sh

### select queue 
#BSUB -q hpc

### name of job, output file and err
#BSUB -J test_loss
#BSUB -o test_loss_%J.out
#BSUB -e test_loss_%J.err

### number of cores
#BSUB -n 1

# request cpu
#BSUB -R "rusage[mem=16G]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

# request 32GB of GPU-memory
#BSUB -R "select[gpu32gb]"

### wall time limit - the maximum time the job will run. Currently 30 min. 
### one modal comb takes longer than 30 min

### this might need to be adjusted 

#BSUB -W 0:30

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
