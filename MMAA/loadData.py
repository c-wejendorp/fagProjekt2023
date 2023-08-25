import numpy as np
from pathlib import Path

trainPath = Path("data/trainingDataSubset")
testPath = Path("data/testDataSubset")

class Real_Data:
    # the default subjects are just a range over all subjects
    # you provide a list of subjects indices if you want to use a subset of subjects
    def __init__(self, concatenation_type: str, subjects = range(1, 17), split = 0):
        """
        concatenation_type (str): "spatial" or "multicondition"
        subjects: list type with subject numbers
        split (int): data split number
        """
        
        # load the data
        subjects = ["sub-{:02d}".format(i) for i in subjects]
        conditions = ["famous", "scrambled", "unfamiliar"]   
             
        EEG_data = []
        MEG_data = []
        fMRI_data = []
        
        #load the data and append to the lists
        for _, subject in enumerate(subjects):            
            if split != 0 and split != 1:
                raise ValueError("only 0 or 1 is valid for the split argument")
            
            EEG_condition = []
            MEG_condition = []
            for condition in conditions:
                # load files depending on arguments given
                EEG_condition.append(np.load((trainPath if split == 0 else testPath) / f"{subject}/eeg/{condition}_train.npy"))
                MEG_condition.append(np.load((trainPath if split == 0 else testPath) / f"{subject}/meg/{condition}_train.npy"))
            
            
            if concatenation_type == np.char.lower("multicondition"):
                # data stored as [s, c, t, v]
                EEG_data.append(np.asarray(EEG_condition))
                MEG_data.append(np.asarray(MEG_condition))
            
            elif concatenation_type == np.char.lower("spatial"):
                # data stored as [s, t, v*3]
                EEG_data.append(np.concatenate(EEG_condition, axis = 1))
                MEG_data.append(np.concatenate(MEG_condition, axis = 1))
            
            else:
                raise ValueError("only multicondition or spatial are valid arguments")
           
            # fMRI data is the same for both splits:
            fmri_temp = np.load(trainPath / f"{subject}/fMRI_train.npy")            

            # some addtional preprocessing of FMRI data
            # for some reason for subject 15, the fMRI data is 1 time point longer.
            # we remove that time point
            if subject == "sub-15":
                fmri_temp = fmri_temp[:-1]  
           
            # for subject 10 the last fMRI run is 170 timesteps instead of 208
            # we extend the fMRI data with zeroes to make it the same length as the other runs
            if subject == "sub-10":
                fmri_temp = np.concatenate((fmri_temp, np.zeros((38, fmri_temp.shape[1]))), axis=0)
                
            
            # since fmri is not split up on conditions, we'll keep the whole fmri time series and duplicate them across conditions
            if concatenation_type == "multicondition":
                fmri_arr = np.array([fmri_temp, fmri_temp, fmri_temp])
                
            elif concatenation_type == "spatial":
                # duplicate the fmri data along the spatial axis (axis 1)
                fmri_arr = np.concatenate([fmri_temp, fmri_temp, fmri_temp], axis = 1)
            else:
                raise ValueError("only multicondition or spatial are valid arguments")
                
            fMRI_data.append(fmri_arr)

        #convert the lists to numpy arrays and make them instance variables to the class
        self.eeg_data = np.asarray(EEG_data)
        self.meg_data = np.asarray(MEG_data)           
        self.fmri_data = np.asarray(fMRI_data)

        #make the instance variables accessible with the old name
        self.EEG_data = self.eeg_data
        self.MEG_data = self.meg_data         
        self.fMRI_data = self.fmri_data  
        
        self.concatenation_type = concatenation_type
        
if __name__ == "__main__":
    X = Real_Data(concatenation_type="spatial", subjects=range(10,16), split=0)
    print(X.eeg_data.shape)
    print(X.meg_data.shape)
    print(X.fmri_data.shape)
    