import json
from createArguments_evening_run import createArguments_evening_run
def createSubmitScripts():
    # read the arguments
    createArguments_evening_run()
    #loop over split
    for splitNum in range(2):
        # loop over the argument files
        for argumentsNum in [0,1]:
            # create the script
            script_template = '''
            #!/bin/sh            


            ### select queue 
            #BSUB -q gpuv100

            ### name of job, output file and err
            #BSUB -J evening_run{split}-{argNum}
            #BSUB -o evening_run{split}-{argNum}_%J.out
            #BSUB -e evening_run{split}-{argNum}_J.err


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

            python MMAA/trainModels_evening_run.py {split} {argNum}
            '''

            script_content = script_template.format(split=splitNum, argNum=argumentsNum)
            with open(f'MMAA/HPC/submit_scripts_evening_run/submit_MMAA_GPU_split-{splitNum}_arg_num-{argumentsNum}.sh', 'w') as fp:
                fp.write(script_content)

if __name__ == "__main__":
    createSubmitScripts()

            










    
    
