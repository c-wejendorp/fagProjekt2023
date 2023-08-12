import numpy as np
import os 
from pathlib import Path

trainPath = Path("data/trainingDataSubset")
testPath = Path("data/testDataSubset")

class Real_Data_oldway:
    # the default subjects are just a range over all subjects
    # you provide a list of subjects indices if you want to use a subset of subjects
    def __init__(self,subjects=range(1, 17), split=0):
        # load the data
        subjects = ["sub-{:02d}".format(i) for i in subjects]
        conditions = ["famous", "scrambled", "unfamiliar"]        
        EEG_data = []
        MEG_data = []
        fMRI_data = []
        #load the data and append to the lists
        for idx,subject in enumerate(subjects):            
            if split == 0:
                EEG_condition = []
                MEG_condition = []
                for condition in conditions:
                    EEG_condition.append(np.load(trainPath / f"{subject}/eeg/{condition}_train.npy"))
                    MEG_condition.append(np.load(trainPath / f"{subject}/meg/{condition}_train.npy"))
                EEG_data.append(np.concatenate(EEG_condition))
                MEG_data.append(np.concatenate(MEG_condition))
                
            elif split ==1:
                EEG_condition = []
                MEG_condition = []
                for condition in conditions:
                    EEG_condition.append(np.load(testPath / f"{subject}/eeg/{condition}_test.npy"))
                    MEG_condition.append(np.load(testPath / f"{subject}/meg/{condition}_test.npy"))
                EEG_data.append(np.concatenate(EEG_condition))
                MEG_data.append(np.concatenate(MEG_condition))      
           
            #fMRI data is the same for both splits:            
            fMRI_data.append(np.load(trainPath / f"{subject}/fMRI_train.npy"))

            # some addtional preprocessing of FMRI data
            # for some reason for subject 15, the fMRI data is 1 time point longer.
            # we remove that time point
            if subject == "sub-15":
                fMRI_data[idx] = fMRI_data[idx][:-1]          
           
            # for subject 10 the last fMRI run is 170 timesteps instead of 208
            # we extend the fMRI data with zeroes to make it the same length as the other runs
            if subject == "sub-10":
                fMRI_data[idx] = np.concatenate((fMRI_data[idx], np.zeros((38, fMRI_data[idx].shape[1]))), axis=0)


        #convert the lists to numpy arrays
        self.eeg_data = np.array(EEG_data)
        self.meg_data = np.array(MEG_data)           
        self.fmri_data = np.array(fMRI_data)

        #make the attribute accessible with the old name

        self.EEG_data = self.eeg_data
        self.MEG_data = self.meg_data         
        self.fMRI_data = self.fmri_data  
          
if __name__ == "__main__":
    X = Real_Data_oldway()
    print(X.eeg_data.shape)
    print(X.meg_data.shape)
    print(X.fmri_data.shape)

    