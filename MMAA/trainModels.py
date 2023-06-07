import sys
import getopt
from loadData import Real_Data
from MMA_model_NEW import MMAA, trainModel
import ast
import numpy as np

# I dont think i actually need this function
"""
def parseArgs(argv):
    # Check if at least one argument is provided
    if len(sys.argv) != 8:
        print("Please provide the following arguments in this specfic order <subjectIdxList> <archeTypeIntevalStart> <archeTypeIntevalStop> <archeTypeStepSize> <seed> <lossRobust> <split>")
        sys.exit(1)

    # Access the command-line arguments
    arguments = {
    'subjects': ast.literal_eval(sys.argv[1]),
    'iterations': sys.argv[2],
    'archeTypeIntevalStart': sys.argv[3],
    'archeTypeIntevalStop': sys.argv[4],
    'archeTypeStepSize': sys.argv[5],
    'seed': sys.argv[6],
    'lossRobust': sys.argv[7],
    'split': sys.argv[8]
                         }

    return arguments
""" 
if __name__ == "__main__":   
        
    # i will implement that we read this from json file later 
    arguments = {
    #'subjects': range(1, 17),
    'subjects': range(1,3),
    'iterations': 100,
    'archeTypeIntevalStart': 14,
    'archeTypeIntevalStop': 16,
    'archeTypeStepSize': 2,
    'seed': 0,
    'lossRobust': True,
    'split': 0                             }
        
    X = Real_Data(subjects=arguments.get("subjects"),split=arguments.get("split"))
    for numArcheTypes in range(arguments.get("archeTypeIntevalStart"),arguments.get("archeTypeIntevalStop")+1, arguments.get("archeTypeStepSize")):
        print(f"Training model with {numArcheTypes} archetypes")
        C, S, eeg_loss, meg_loss, fmri_loss, loss_Adam = trainModel(
                X,
                numArchetypes=numArcheTypes,
                seed=arguments.get("seed"),
                plotDistributions=False,
                learningRate=1e-1,
                numIterations=arguments.get("iterations"), 
                loss_robust=arguments.get("lossRobust"))
        
        seed = arguments.get("seed")

        np.save(f'MMAA/modelsInfo/C_matrix_k{numArcheTypes}_s{seed}', C)
        np.save(f'MMAA/modelsInfo/S_matrix_k{numArcheTypes}_s{seed}', S)
        np.save(f'MMAA/modelsInfo/eeg_loss{numArcheTypes}_s{seed}', np.array([int(x.detach().numpy())for x in eeg_loss])) 
        np.save(f'MMAA/modelsInfo/meg_loss{numArcheTypes}_s{seed}', np.array([int(x.detach().numpy())for x in meg_loss]))
        np.save(f'MMAA/modelsInfo/fmri_loss{numArcheTypes}_s{seed}', np.array([int(x.detach().numpy())for x in fmri_loss]))
        np.save(f'MMAA/modelsInfo/loss_adam{numArcheTypes}_s{seed}', np.array(loss_Adam))
      
 