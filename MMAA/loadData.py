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
            EEG_data.append(np.load(f"data/trainingDatasSubset/{subject}/eeg.npy"))
            MEG_data.append(np.load(f"data/trainingDatasSubset/{subject}/meg.npy"))
            fMRI_data.append(np.load(f"data/trainingDatasSubset/{subject}/fmri.npy"))
        #convert the lists to numpy arrays
        self.EEG_data = np.array(EEG_data)
        self.MEG_data = np.array(MEG_data)
        # for some reason for subject 15, the fMRI data is 1 time point longer.
        # we remove that time point
        if numSubjects >= 15:
            fMRI_data[14] = fMRI_data[14][:-1]   #remember that we have zero indexing
        self.fMRI_data = np.array(fMRI_data) 
          

    