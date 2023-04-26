import torch 
import numpy as np
import matplotlib.pyplot as plt
from MMAA_model import MMAA

numSources = 20484 
numConditions = 3
timeSteps = 180

#XMS.shape = (Modalaity, Subject, Time * numConditions , numSources * numConditions)
Xms = np.zeros((2,2, timeSteps * numConditions, numSources * numConditions ))
for modalityIDX, modality in enumerate(["meg", "eeg"]):
    for subjectIDX, subject in enumerate(["sub-01", "sub-02"]):        
        with open(f'data/trainingDatasSubset/{subject}/{modality}.npy', 'rb') as f:            
            Xms[modalityIDX, subjectIDX, :, :] = np.load(f)
            print(2)

#print(Xms)
print(2)
 