
            #!/bin/sh            


            ### select queue 
            #BSUB -q gpuv100

            ### name of job, output file and err
            #BSUB -J MMAA_train_split-0_arg_num-3
            #BSUB -o MMAA_train_split-0_arg_num-3_%J.out
            #BSUB -e MMAA_train_split-0_arg_num-3_J.err


            ### number of cores
            #BSUB -n 1

            # request cpu
            #BSUB -R "rusage[mem=16G]"

            ### -- Select the resources: 1 gpu in exclusive process mode --
            #BSUB -gpu "num=1:mode=exclusive_process"

            # request 32GB of GPU-memory
            #BSUB -R "select[gpu32gb]"

            ### wall time limit - the maximum time the job will run. Currently 3.5 hours. 

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

            python MMAA/trainModels.py 0 3
            