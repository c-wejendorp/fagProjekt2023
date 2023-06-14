import json

def createSubmitScripts():
    # read the arguments    
    #loop over arhetypes
    for numArch in range(2,40+1,2):
            script_template = """

            #!/bin/sh
            ### select queue 
            #BSUB -q hpc

            ### name of job, output file and err
            #BSUB -J Classifier_k-{k}
            #BSUB -o Classifier_k-{k}_%J.out
            #BSUB -e Classifier_k-{k}_%J.err


            ### number of cores
            #BSUB -n 1

            ### -- specify that the cores must be on the same host -- 
            #BSUB -R "span[hosts=1]"
            ### -- specify that we need 8GB of Memory  more than enough for this task
            #BSUB -R "rusage[mem=8GB]"

            ### wall time limit - the maximum time the job will run. Currently 2 hours  min. 

            #BSUB -W 02:00

            ##BSUB -u s204090@dtu.dk
            ### -- send notification at start -- 
            #BSUB -B 
            ### -- send notification at completion -- 
            #BSUB -N 

            # end of BSUB options


            # load the correct  scipy module and python

            module load scipy/1.10.1-python-3.11.3
            #module load cuda/11.8


            # activate the virtual environment
            # NOTE: needs to have been built with the same SciPy version above!
            source MMAA/HPC_env/bin/activate

            python Classifier/plot_results_HPC/plot_results_k-{k}.py"""          

            script_content = script_template.format(k=numArch)
            with open(f'Classifier/submit_scripts/submit_Classifier_k-{numArch}.sh', 'w') as fp:
            
                fp.write(script_content)

if __name__ == "__main__":
    createSubmitScripts()

            










    
    
