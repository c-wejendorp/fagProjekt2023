from pathlib import Path
import mne
import pandas as pd

# DEBUG PURPORSE FILE; this file creates a csv file with an overview of the subjects 
# and the number of trials in each condition after preprocessings

#path to the freesurfer directory
fs_dir = Path("data/freesurfer")

#path to the data directory (JespersProcessed)
data_dir = Path("data/JesperProcessed")

#create dataframe for overview of subjects
df = pd.DataFrame(columns=["subject", "famous", "unfamiliar", "scrambled","total"])
for subject_id in range(1,17):
    #assign paths
    subject = f"sub-{subject_id:02d}"
    subject_dir = data_dir / subject
    meg_dir = subject_dir / "ses-meg"
    fwd_dir = meg_dir / "stage-forward"
    pre_dir = meg_dir / "stage-preprocess"
    inv_dir = meg_dir / "stage-inverse"
    mri_dir = subject_dir / "ses-mri"
    fmri_dir = mri_dir / "func"

    # read all epocs and add rows to dataframe
    epo = mne.read_epochs(pre_dir / "task-facerecognition_proc-p_epo.fif")
    df.loc[subject_id] = [subject_id, len(epo["famous"]), len(epo["unfamiliar"]), len(epo["scrambled"]),sum([len(epo["famous"]), len(epo["unfamiliar"]), len(epo["scrambled"])])]

# save dataframe to csv file
df.to_csv("data/subject_overview.csv")


