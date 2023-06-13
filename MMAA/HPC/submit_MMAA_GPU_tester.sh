
            #!/bin/sh            


            ### select queue 
            #BSUB -q hpc

            ### name of job, output file and err
            #BSUB -J MMAA_train_test
            #BSUB -o MMAA_train_test_%J.out
            #BSUB -e MMAA_train_test-0_J.err


            ### number of cores
            #BSUB -n 1

            # request cpu
            #BSUB -R "rusage[mem=32G]"

            ### we dont need gpu for the test. 

           

            ### wall time limit - the maximum time the job will run. For this tester only 5 min. 

            #BSUB -W 00:10

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

            python MMAA/trainModels_tester.py
            