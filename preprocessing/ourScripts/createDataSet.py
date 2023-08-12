from pathlib import Path
import mne
import numpy as np
from scipy.signal import butter, filtfilt
from checkPath import checkPath
checkPath()
from projects.facerecognition_dtu import utils
from projects.facerecognition_dtu.config import Config

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype="highpass")
    
    return b, a

def butter_highpass_filter(data, lowcut=0.008, fs=1/2, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)

    return y

# Output paths
trainPath = Path("data/trainingDataSubset")
testPath = Path("data/testDataSubset")

#this file creates a dataset for each subject
# needs to be updates such that FMRI is handled correctly regarding mean subtraction.

# For individual EPOCHs and ERP see notes on data in the folder JesperScripts
# We want to load the source space data starting with MEG and EEG

def EEG_AND_MEG(subject,data_dir="data/JesperProcessed",split=0):
    """ Function that saves the EEG and MEG data in .npy format in the respective test and train folders. 
    In addition, the corpus callosum sources are removed and the source time series are frobenius normed.
    
    subject (str): Format; sub-{:02d} Example: sub-02
    data_dir (str): data directory with all the preprocessed subjects
    split (int): data split number
    """
    
    # assign paths
    data_dir = Path(data_dir) 
    fs_dir = Path("data/freesurfer")
    
    subject_dir = data_dir / subject
    meg_dir = subject_dir / "ses-meg"    
    inv_dir = meg_dir / "stage-inverse" 
    
    # label indices for corpus callosum in each hemisphere (888 for lh, 881 for rh)
    label_lh = mne.read_label(fs_dir / "fsaverage/label/lh.Medial_wall.label",subject=subject)
    label_rh = mne.read_label(fs_dir / "fsaverage/label/rh.Medial_wall.label",subject=subject)
    label_both = np.concatenate((label_lh.vertices, label_rh.vertices + 10242)) #we add 10242 because that marks the beginning of the right hemisphere

    for modality in ["meg", "eeg"]:
        for condition in ["famous", "unfamiliar", "scrambled"]:
            # shape before removing corpus callosum: [10242, 10242]
            fsaverageSources = mne.read_source_estimate(inv_dir / f"task-facerecognition_space-fsaverage_cond-{condition}_fwd-mne_ch-{modality}_split-{split}_stc")
            
            # remove corpus callosum sources for both hemispheres - both vertices and the data itself
            # shape after removing: [9354, 9361]
            fsaverageSources.vertices[0] = np.delete(fsaverageSources.vertices[0], label_lh.vertices)
            fsaverageSources.vertices[1] = np.delete(fsaverageSources.vertices[1], label_rh.vertices)
            fsaverageSources.data = np.delete(fsaverageSources.data, label_both, axis = 0)
            
            # save data [source x time] as new variable we can tamper with
            sourceTimesSeries = fsaverageSources.data 
            sourceTimesSeries = sourceTimesSeries.T
            
            # normalizing the data using frobenius normalization 
            frobeniusNorm = np.linalg.norm(sourceTimesSeries, ord='fro')
            normalizedSourceTimesSeries= sourceTimesSeries / frobeniusNorm
            
            # save the splits as separate files
            if split == 0:
                with open(trainPath / f'{subject}/{modality}/{condition}_train.npy', 'wb') as f:
                    np.save(f, normalizedSourceTimesSeries)
            elif split == 1:
                with open(testPath / f'{subject}/{modality}/{condition}_test.npy', 'wb') as f:
                    np.save(f, normalizedSourceTimesSeries)
       
def fMRI(subject, data_dir="data/JesperProcessed"):
    """ Function that saves the data in .npy format in the respective test and train folders.
    In addition, it removes the corpus callosum sources and frobenius norms the source time series.
    Do take note that there are no splits, so we use the full fmri data for training and testing
    
    subject (str): Format; sub-{:02d} Example: sub-02
    data_dir (str): data directory with all the preprocessed subjects
    """
    
    # create new morpher from the entire source space (from Jesper)
    for subject_id in range(1,17):
        io = utils.SubjectIO(subject_id)
        src = mne.read_source_spaces(io.data.get_filename(stage="forward", forward="mne", suffix="src"))
        fmri_morph = mne.compute_source_morph(src, subjects_dir=Config.path.FREESURFER)
        
        #create a folder if you don't have one
        Path("data/fmriMorphers").mkdir(parents=True, exist_ok=True)
        fmri_morph.save(f"data/fmriMorphers/sub-{subject_id:02d}", overwrite=True)    
    
    # assign paths
    data_dir = Path(data_dir)
    fs_dir = Path("data/freesurfer")
    
    subject_dir = data_dir / subject    
    mri_dir = subject_dir / "ses-mri"
    fmri_dir = mri_dir / "func"
    
    # label indices for corpus callosum in each hemisphere (888 for lh, 881 for rh)
    label_lh = mne.read_label(fs_dir / "fsaverage/label/lh.Medial_wall.label",subject=subject)
    label_rh = mne.read_label(fs_dir / "fsaverage/label/rh.Medial_wall.label",subject=subject)
    label_both = np.concatenate((label_lh.vertices, label_rh.vertices + 10242)) #we add 10242 because that marks the beginning of the right hemisphere
    

    fMRIdata = []
    runLengths = []
    for run in ["run-{:02d}".format(i) for i in range(1, 10)]: 
        # read sources, collect correct morpher and morph into fsaverage
        FMRIstc = mne.read_source_estimate(fmri_dir / f"surf_sa{subject}_ses-mri_task-facerecognition_{run}_bold")
        FMRImorph = mne.read_source_morph(f"data/fmriMorphers/{subject}-morph.h5")
        FMRIstc_morphed=FMRImorph.apply(FMRIstc)
        
        # highpass filter the fmri signal
        FMRIstc_morphed.data = butter_highpass_filter(FMRIstc_morphed.data, lowcut = 0.008)
        
        # remove corpus callosum sources for both hemispheres
        # shape before removing corpus callosum: [10242, 10242]
        # shape after removing: [9354, 9361]
        FMRIstc_morphed.vertices[0] = np.delete(FMRIstc_morphed.vertices[0], label_lh.vertices)
        FMRIstc_morphed.vertices[1] = np.delete(FMRIstc_morphed.vertices[1], label_rh.vertices)
        FMRIstc_morphed.data = np.delete(FMRIstc_morphed.data, label_both, axis = 0)
            
        # save data [source x time] as new variable we can tamper with
        sourceTimesSeries = FMRIstc_morphed.data 
        sourceTimesSeries = sourceTimesSeries.T
        
        # subtracting mean value
        fMRI_mean = np.mean(sourceTimesSeries, axis = 0)
        sourceTimesSeries -= fMRI_mean
        
        # normalizing the data using frobenius normalization 
        frobeniusNorm = np.linalg.norm(sourceTimesSeries, ord='fro')
        normalizedSourceTimesSeries= sourceTimesSeries / frobeniusNorm
        
        # concatanate the data from the different runs
        fMRIdata.append(normalizedSourceTimesSeries)

        # DEBUG PURPOSE: Check run lengths
        runLengths.append(normalizedSourceTimesSeries.shape[0])
    
    # concatenate the runs
    fMRIdata = np.concatenate(fMRIdata) 

    # save as files
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
        EEG_AND_MEG(subject, split=0)
        EEG_AND_MEG(subject, split=1)
        fMRI(subject)


    
   
    

