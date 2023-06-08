from pathlib import Path
import mne
import numpy as np

trainPath = Path("data/trainingDataSubset")
testPath = Path("data/testDataSubset")

#this file creates a dataset for each subject
# needs to be updates such that FMRI is handled correctly regarding mean subtraction.

# For individual EPOCHs and ERP see notes on data in the folder JesperScripts
# We want to load the source space data starting with MEG and EEG

def EEG_AND_MEG(subject,data_dir="data/JesperProcessed",split=0):
    data_dir = Path(data_dir) 
    fs_dir = Path("data/freesurfer")
    
    #label indices for corpus callosum in each hemisphere (888 for lh, 881 for rh)
    label_lh = mne.read_label(fs_dir / "fsaverage/label/lh.Medial_wall.label",subject=subject)
    label_rh = mne.read_label(fs_dir / "fsaverage/label/rh.Medial_wall.label",subject=subject)
    label_both = np.concatenate((label_lh.vertices, label_rh.vertices + 10242))
    
    subject_dir = data_dir / subject
    meg_dir = subject_dir / "ses-meg"    
    inv_dir = meg_dir / "stage-inverse"    

    for modality in ["meg", "eeg"]:
        # conditionTimeSeries = [] 
        for condition in ["famous", "unfamiliar", "scrambled"]:
            #shape before removing corpus callosum: [10242, 10242]
            fsaverageSources = mne.read_source_estimate(inv_dir / f"task-facerecognition_space-fsaverage_cond-{condition}_fwd-mne_ch-{modality}_split-{split}_stc")
            
            #remove corpus callosum sources for both hemispheres - both vertices and the data itself
            #shape after removing: [9354, 9361]
            fsaverageSources.vertices[0] = np.delete(fsaverageSources.vertices[0], label_lh.vertices)
            fsaverageSources.vertices[1] = np.delete(fsaverageSources.vertices[1], label_rh.vertices)
            fsaverageSources.data = np.delete(fsaverageSources.data, label_both, axis = 0)
            
            #as numpy array 
            sourceTimesSeries = fsaverageSources.data 
            # transpose to get dimension (time, source)
            sourceTimesSeries = sourceTimesSeries.T
            # normalizing the data using frobenius normalization 
            frobeniusNorm = np.linalg.norm(sourceTimesSeries, ord='fro')
            # remember this normalization normalize the norm of the matrix
            # this means that each "activation" does not lie btw 0 and 1 as we also have discussed
            normalizedSourceTimesSeries= sourceTimesSeries / frobeniusNorm
            # conditionTimeSeries.append(normalizedSourceTimesSeries)
            
            if split == 0:
                with open(trainPath / f'{subject}/{modality}/{condition}_train.npy', 'wb') as f:
                    np.save(f, normalizedSourceTimesSeries)
            elif split == 1:
                with open(testPath / f'{subject}/{modality}/{condition}_test.npy', 'wb') as f:
                    np.save(f, normalizedSourceTimesSeries)
            
        # # concatenate the list of arrays to one array corresponding to concatenating the three condtions  
        # conditionTimeSeries = np.concatenate(conditionTimeSeries)
        # if split == 0:
        #     with open(trainPath / f'{subject}/{modality}_train.npy', 'wb') as f:
        #         np.save(f, conditionTimeSeries)
        # elif split == 1:
        #     with open(testPath / f'{subject}/{modality}_test.npy', 'wb') as f:
        #         np.save(f, conditionTimeSeries)

# now do the same for the fMRI data, here the splits doesnt matter, as we use the full fmri data for training and testing
def fMRI(subject, data_dir="data/JesperProcessed", morpherFolder = "data/fmriMorphers"):   
    data_dir = Path(data_dir)
    fs_dir = Path("data/freesurfer")
    
    subject_dir = data_dir / subject    
    mri_dir = subject_dir / "ses-mri"
    fmri_dir = mri_dir / "func"
    
    #label indices for corpus callosum in each hemisphere (888 for lh, 881 for rh)
    label_lh = mne.read_label(fs_dir / "fsaverage/label/lh.Medial_wall.label",subject=subject)
    label_rh = mne.read_label(fs_dir / "fsaverage/label/rh.Medial_wall.label",subject=subject)
    label_both = np.concatenate((label_lh.vertices, label_rh.vertices + 10242))
    
    # we use all the runs of the fMRI data as training data
    # we will use the last run as test data
    fMRIdata = []
    runLengths = []
    for run in ["run-{:02d}".format(i) for i in range(1, 10)]: 
        # indlæsning af fMRI data i fsaverage space gøres således:
        #shape before removing corpus callosum: [10242, 10242]
        FMRIstc = mne.read_source_estimate(fmri_dir / f"surf_sa{subject}_ses-mri_task-facerecognition_{run}_bold")
        # Hent korrekt morpher
        FMRImorph = mne.read_source_morph(f"data/fmriMorphers/{subject}-morph.h5")
        # Morph fMRI data til fsaverage space
        FMRIstc_morphed=FMRImorph.apply(FMRIstc)
        
        #remove corpus callosum sources for both hemispheres
        #shape after removing: [9354, 9361]
        FMRIstc_morphed.vertices[0] = np.delete(FMRIstc_morphed.vertices[0], label_lh.vertices)
        FMRIstc_morphed.vertices[1] = np.delete(FMRIstc_morphed.vertices[1], label_rh.vertices)
        FMRIstc_morphed.data = np.delete(FMRIstc_morphed.data, label_both, axis = 0)
            
        #as numpy array 
        sourceTimesSeries = FMRIstc_morphed.data 
        # transpose to get dimension (time, source)
        sourceTimesSeries = sourceTimesSeries.T
        
        #subtracting mean value
        fMRI_mean = np.mean(sourceTimesSeries, axis = 0)
        sourceTimesSeries -= fMRI_mean
        
        # normalizing the data using frobenius normalization 
        frobeniusNorm = np.linalg.norm(sourceTimesSeries, ord='fro')
        # remember this normalization normalize the norm of the matrix
        # this means that each "activation" does not lie btw 0 and 1 as we also have discussed
        normalizedSourceTimesSeries= sourceTimesSeries / frobeniusNorm
        # concatanate the data from the different runs
        fMRIdata.append(normalizedSourceTimesSeries)
        #print("The shape of the fMRI data is: ", FMRIstc_morphed.data.shape)

        #check if all runs have the same lenght
        runLengths.append(normalizedSourceTimesSeries.shape[0])
       
    #print(runLengths)    
    fMRIdata = np.concatenate(fMRIdata)

    with open(trainPath / f'{subject}/fMRI_train.npy', 'wb') as f:
        np.save(f, fMRIdata)
    
    #with open(testPath / f'{subject}/fMRI_test.npy', 'wb') as f:
    #    np.save(f, fMRIdata) 
    

if __name__ == "__main__":
    # create a list of all subjects with 1 leading zero
    subjects = ["sub-{:02d}".format(i) for i in range(1, 17)]
    eegmeg = ["eeg", "meg"]
    # loop over all subjects
    for subject in subjects:
        for modality in eegmeg:
            # create a folder for each subject
            Path(trainPath /f"{subject}" / f"{modality}").mkdir(parents=True, exist_ok=True)
            Path(testPath /f"{subject}" / f"{modality}").mkdir(parents=True, exist_ok=True)        
        # create the data for each subject
        EEG_AND_MEG(subject,split=0)
        EEG_AND_MEG(subject,split=1)
        fMRI(subject)


    
   
    

