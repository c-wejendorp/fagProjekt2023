import json
import sys
import getopt
from loadData import Real_Data
from MMA_model_CUDA import MMAA, trainModel
import ast
import numpy as np


if __name__ == "__main__":  
    #load arguments from json file
    with open('MMAA/arguments.json') as f:
        arguments = json.load(f)    
        
    X = Real_Data(subjects=arguments.get("subjects"),split=arguments.get("split"))
    # loop over seeds
    for seed in arguments.get("seeds"):
        for numArcheTypes in range(arguments.get("archeTypeIntevalStart"),arguments.get("archeTypeIntevalStop")+1, arguments.get("archeTypeStepSize")):
            print(f"Training model with {numArcheTypes} archetypes and seed {seed}")
            C, S, eeg_loss, meg_loss, fmri_loss, loss_Adam = trainModel(
                    X,
                    numArchetypes=numArcheTypes,
                    seed=seed,
                    plotDistributions=False,
                    learningRate=1e-1,
                    numIterations=arguments.get("iterations"), 
                    loss_robust=arguments.get("lossRobust"))          
            split = arguments.get("split")
            np.save(f'MMAA/modelsInfo/C_matrix_k{numArcheTypes}_s{seed}_split{split}', C)
            np.save(f'MMAA/modelsInfo/S_matrix_k{numArcheTypes}_s{seed}_split{split}', S)
            np.save(f'MMAA/modelsInfo/eeg_loss{numArcheTypes}_s{seed}_split{split}', np.array([int(x.cpu().detach().numpy())for x in eeg_loss])) 
            np.save(f'MMAA/modelsInfo/meg_loss{numArcheTypes}_s{seed}_split{split}', np.array([int(x.cpu().detach().numpy())for x in meg_loss]))
            np.save(f'MMAA/modelsInfo/fmri_loss{numArcheTypes}_s{seed}_split{split}', np.array([int(x.cpu().detach().numpy())for x in fmri_loss]))
            np.save(f'MMAA/modelsInfo/loss_adam{numArcheTypes}_s{seed}_split{split}', np.array(loss_Adam))