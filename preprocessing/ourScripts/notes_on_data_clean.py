from pathlib import Path
import mne

#path to the freesurfer directory
#fs_dir = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition/freesurfer")
fs_dir = Path("data/freesurfer")

#path to the data directory (JespersProcessed)
#data_dir = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition_dtu/data")
data_dir = Path("data/JesperProcessed")
subject = "sub-02"

subject_dir = data_dir / subject
meg_dir = subject_dir / "ses-meg"
fwd_dir = meg_dir / "stage-forward"
pre_dir = meg_dir / "stage-preprocess"
inv_dir = meg_dir / "stage-inverse"

mri_dir = subject_dir / "ses-mri"
fmri_dir = mri_dir / "func"

# Read sensor space data

# Epochs
epo = mne.read_epochs(pre_dir / "task-facerecognition_proc-p_epo.fif")
# epo["famous"] to get epochs for `famous` condition
# also epo.plot(), epo.get_data()

# ERP
evo = mne.read_evokeds(pre_dir / "task-facerecognition_proc-p_cond-famous_split-0_evo.fif")
evo = evo[0]
# evo.plot(), evo.get_data()

# orginal notes/code from Jesper below
"""
# Read source space data
# to get from subject space to fsaverage
# without -[l/r]h.stc !
stc = mne.read_source_estimate("task-facerecognition_cond-famous_fwd-mne_ch-eeg_split-0_stc")
morph = mne.read_source_morph(fwd_dir / "task-facerecognition_fwd-mne_morph.h5")
morph.apply(stc)
# I already did this; should be equal to
stc = mne.read_source_estimate("task-facerecognition_space-fsaverage_cond-famous_fwd-mne_ch-eeg_split-0_stc")
# get data: stc.data, stc.lh_data, stc.rh_data

# fMRI is read in a similar manner!
# I set time step to 2000 ms which is also repetition time (TR) so timings
# should be valid
# stimuli timing and identity is in `fmri_dir` .tsv files
"""

#_________________________________________________________________________________________________________________________________#
# Vi skal bruge det allerede morphede M/EEG data, da disse har korrekt shape.
# Vi skal lave nye morph objekter til fMRI data. Disse laves i fmriMorpher

# indlæsning af  M/EEG data i fsaverage space er så simplet som: 
MEGstc_morphed = mne.read_source_estimate(inv_dir / "task-facerecognition_space-fsaverage_cond-famous_fwd-mne_ch-eeg_split-0_stc")

# indlæsning af fMRI data i fsaverage space gøres således:
FMRIstc = mne.read_source_estimate(fmri_dir / f"surf_sa{subject}_ses-mri_task-facerecognition_run-01_bold")
# Hent korrekt morpher
FMRImorph = mne.read_source_morph(f"data/fmriMorphers/{subject}-morph.h5")

# Morph fMRI data til fsaverage space
FMRIstc_morphed=FMRImorph.apply(FMRIstc)
# denne fil skal vi så lige gemme et smart sted !? 

#dobbelt check af shape
print("The shape of the M/EEG data is: ", MEGstc_morphed.data.shape)
print("The shape of the fMRI data is: ", FMRIstc_morphed.data.shape)

# Plotting source space data
# You can *try* this but I am not sure it will work. If it does, will show
# the data on the brain surface and allow you to scroll through in time
# Doesn't play nicely with "dark background" from matplotlib so perhaps do
# plt.style.use("default") if you run in to problems

#mne.viz.plot_source_estimates(MEGstc_morphed,subject=subject, subjects_dir=fs_dir)

#works 
#MEGstc_morphed.plot(subject, subjects_dir=fs_dir)

# doesn't work
#FMRIstc_morphed.plot(subject, subjects_dir=fs_dir)
# does however work on the NONE morphed data
FMRIstc.plot(subject, subjects_dir=fs_dir)

# works
# Show the BEM surfaces used to construct the "forward model"
mne.viz.plot_bem(subject, fs_dir) # brain_surfaces="white"