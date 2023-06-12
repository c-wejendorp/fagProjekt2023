
            #!/bin/sh

            ## should be run as filename.sh <split_number> <argsNum>
            # example submit_MMAA_GPU.sh 0 0


            ### select queue 
            #BSUB -q gpuv100

            ### name of job, output file and err
            #BSUB -J MMAA_train_split-0
            #BSUB -o MMAA_train_split-0_%J.out
            #BSUB -e MMAA_train_split-0_%J.err


            ### number of cores
            #BSUB -n 1

            # request cpu
            #BSUB -R "rusage[mem=16G]"

            ### -- Select the resources: 1 gpu in exclusive process mode --
            #BSUB -gpu "num=1:mode=exclusive_process"

            # request 32GB of GPU-memory
            #BSUB -R "select[gpu32gb]"

            ### wall time limit - the maximum time the job will run. Currently 3.5 hours. 

            #BSUB -W 03:30

            ##BSUB -u s204090@dtu.dk
            ### -- send notification at start -- 
            #BSUB -B 
            ### -- send notification at completion -- 
            #BSUB -N 


            # end of BSUB options

            # Access the command line arguments
            split=$1
            argNum=$2


            # load the correct  scipy module and python

            module load scipy/1.10.1-python-3.11.3
            module load cuda/11.8


            # activate the virtual environment
            # NOTE: needs to have been built with the same SciPy version above!
            source MMAA/HPC_env/bin/activate

            python MMAA/trainModels.py {split} {argNum}
            