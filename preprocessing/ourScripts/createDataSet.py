from pathlib import Path
import mne
import numpy as np

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

# Read sensor space data

# For individual EPOCHs and ERP see notes on data in the folder JesperScripts
# We want to load the source space data starting with MEG and EEG

for modality in ["meg", "eeg"]:
    condtionTimeSeries = [] 
    for condtion in ["famous", "unfamiliar", "scrambled"]:
        fsaverageSources = mne.read_source_estimate(inv_dir / f"task-facerecognition_space-fsaverage_cond-{condtion}_fwd-mne_ch-{modality}_split-0_stc")
        #as numpy array 
        sourceTimesSeries = fsaverageSources.data 
        # normalizing the data using frobenius normalization 
        frobeniusNorm = np.linalg.norm(sourceTimesSeries, ord='fro')
        # remember this normalization normalize the norm of the matrix
        # this means that each "activation" does not lie btw 0 and 1 as we also have discussed
        normalizedSourceTimesSeries= sourceTimesSeries / frobeniusNorm
        condtionTimeSeries.append(normalizedSourceTimesSeries)
    condtionTimeSeries = np.concatenate(condtionTimeSeries)
    with open(f'data/trainingDatasSubset/{subject}/{modality}.npy', 'wb') as f:
        np.save(f, condtionTimeSeries)

# now do the same for the fMRI data
# not done yet.
    

        
                



# indlæsning af  M/EEG data i fsaverage space er så simplet som: 
MEGstc_morphed = mne.read_source_estimate(inv_dir / "task-facerecognition_space-fsaverage_cond-famous_fwd-mne_ch-eeg_split-0_stc")
