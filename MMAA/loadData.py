import numpy as np
import os 
class Real_Data:
    def __init__(self,numSubjects=16):
        # load the data
        subjects = ["sub-{:02d}".format(i) for i in range(1, numSubjects+1)]
        EEG_data = []
        MEG_data = []
        fMRI_data = []
        #load the data and append to the lists
        for subject in subjects:     
            test=2   
            EEG_data.append(np.load(f"data/trainingDatasSubset/{subject}/eeg.npy"))
            MEG_data.append(np.load(f"data/trainingDatasSubset/{subject}/meg.npy"))
            fMRI_data.append(np.load(f"data/trainingDatasSubset/{subject}/fmri.npy"))
        #convert the lists to numpy arrays
        self.EEG_data = np.array(EEG_data)
        self.MEG_data = np.array(MEG_data)
        self.fMRI_data = np.array(fMRI_data) 
          

    