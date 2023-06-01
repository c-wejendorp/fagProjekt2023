from pathlib import Path
import mne
import numpy as np

"""
#path to the freesurfer directory
fs_dir = Path("data/freesurfer")

#path to the data directory (JespersProcessed)
data_dir = Path("data/JesperProcessed")

#from here could be in a loop over all our subjects but for now we just do it manually
subject = "sub-02"

subject_dir = data_dir / subject
meg_dir = subject_dir / "ses-meg"
fwd_dir = meg_dir / "stage-forward"
pre_dir = meg_dir / "stage-preprocess"
inv_dir = meg_dir / "stage-inverse"

mri_dir = subject_dir / "ses-mri"
fmri_dir = mri_dir / "func"
"""
# Read sensor space data

# For individual EPOCHs and ERP see notes on data in the folder JesperScripts
# We want to load the source space data starting with MEG and EEG

def EEG_AND_MEG(subject,data_dir="data/JesperProcessed"):
    data_dir = Path(data_dir)   
    subject_dir = data_dir / subject
    meg_dir = subject_dir / "ses-meg"    
    inv_dir = meg_dir / "stage-inverse"    

    for modality in ["meg", "eeg"]:
        conditionTimeSeries = [] 
        for condtion in ["famous", "unfamiliar", "scrambled"]:
            fsaverageSources = mne.read_source_estimate(inv_dir / f"task-facerecognition_space-fsaverage_cond-{condtion}_fwd-mne_ch-{modality}_split-0_stc")
            #as numpy array 
            sourceTimesSeries = fsaverageSources.data 
            # transpose to get dimension (time, source)
            sourceTimesSeries = sourceTimesSeries.T
            # normalizing the data using frobenius normalization 
            frobeniusNorm = np.linalg.norm(sourceTimesSeries, ord='fro')
            # remember this normalization normalize the norm of the matrix
            # this means that each "activation" does not lie btw 0 and 1 as we also have discussed
            normalizedSourceTimesSeries= sourceTimesSeries / frobeniusNorm
            conditionTimeSeries.append(normalizedSourceTimesSeries)
        # concatenate the list of arrays to one array corresponding to concatenating the three condtions  
        conditionTimeSeries = np.concatenate(conditionTimeSeries)
        with open(f'data/trainingDatasSubset/{subject}/{modality}.npy', 'wb') as f:
            np.save(f, conditionTimeSeries)

# now do the same for the fMRI data
def fMRI(subject, data_dir="data/JesperProcessed", morpherFolder = "data/fmriMorphers"):   
    data_dir = Path(data_dir)
    subject_dir = data_dir / subject
    #meg_dir = subject_dir / "ses-meg"
    #fwd_dir = meg_dir / "stage-forward"
    #pre_dir = meg_dir / "stage-preprocess"
    #inv_dir = meg_dir / "stage-inverse"
    mri_dir = subject_dir / "ses-mri"
    fmri_dir = mri_dir / "func"

    # for now we use the first 5 runs of the fMRI data as training data
    # we will use the last run as test data
    fMRIdata = []
    for run in ["run-{:02d}".format(i) for i in range(1, 6)]: 

        # indlæsning af fMRI data i fsaverage space gøres således:
        FMRIstc = mne.read_source_estimate(fmri_dir / f"surf_sa{subject}_ses-mri_task-facerecognition_{run}_bold")
        # Hent korrekt morpher
        FMRImorph = mne.read_source_morph(f"data/fmriMorphers/{subject}-morph.h5")
        # Morph fMRI data til fsaverage space
        FMRIstc_morphed=FMRImorph.apply(FMRIstc)    
        #as numpy array 
        sourceTimesSeries = FMRIstc_morphed.data 
        # transpose to get dimension (time, source)
        sourceTimesSeries = sourceTimesSeries.T
        # normalizing the data using frobenius normalization 
        frobeniusNorm = np.linalg.norm(sourceTimesSeries, ord='fro')
        # remember this normalization normalize the norm of the matrix
        # this means that each "activation" does not lie btw 0 and 1 as we also have discussed
        normalizedSourceTimesSeries= sourceTimesSeries / frobeniusNorm
        # concatanate the data from the different runs
        fMRIdata.append(normalizedSourceTimesSeries)
        #print("The shape of the fMRI data is: ", FMRIstc_morphed.data.shape)

    fMRIdata = np.concatenate(fMRIdata)

    with open(f'data/trainingDatasSubset/{subject}/fMRI.npy', 'wb') as f:
        np.save(f, fMRIdata)

if __name__ == "__main__":
    # create a list of all subjects with 1 leading zero
    subjects = ["sub-{:02d}".format(i) for i in range(1, 17)]
    # loop over all subjects
    for subject in subjects:
        # create a folder for each subject
        Path(f"data/trainingDatasSubset/{subject}").mkdir(parents=True, exist_ok=True)
        # create the data for each subject
        EEG_AND_MEG(subject)
        fMRI(subject)


    
   
    

