from pathlib import Path
import mne

# 
# this file is basically just notes on how to load the data and visualize it
# the "orignal" can be found as notes_on_data in the folder JesperScripts

#path to the freesurfer directory
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
#_________________________________________________________________________________________________________________________________#
# Vi skal bruge det allerede morphede M/EEG data, da disse har korrekt shape.
# Vi skal lave nye morph objekter til fMRI data. Disse laves i fmriMorpher

# indlæsning af  M/EEG data i fsaverage space er så simplet som: 
MEGstc_morphed = mne.read_source_estimate(inv_dir / "task-facerecognition_space-fsaverage_cond-famous_fwd-mne_ch-eeg_split-0_stc")

# indlæsning af fMRI data i fsaverage space gøres således:
FMRIstc = mne.read_source_estimate(fmri_dir / f"surf_sa{subject}_ses-mri_task-facerecognition_run-01_bold")
# Hent korrekt morpher, denne laves i fmriMorpher.py
FMRImorph = mne.read_source_morph(f"data/fmriMorphers/{subject}-morph.h5")

# Morph fMRI data til fsaverage space
FMRIstc_morphed=FMRImorph.apply(FMRIstc)

#dobbelt check af shape
print("The shape of the M/EEG data is: ", MEGstc_morphed.data.shape)
print("The shape of the fMRI data is: ", FMRIstc_morphed.data.shape)

# lille note i forhold visualisering af data
# når man morpher direkte i scriptet, som der gøres her med FMRI, så skal subject ikke angives i plotfunktionen
# men hvis man derimod henter et morph objekt fra en fil, så skal subject angives som "fsaverage" i plot funktionen

# Indlæs labels fra fsaverage
labels = mne.read_labels_from_annot("fsaverage", parc="aparc_sub", subjects_dir=fs_dir)

# loop over every other label and plot it
for label in labels[::2]:
    print("plotting:")
    print(label.name)
    try:
        region_stc = MEGstc_morphed.in_label(label)
        region_plot = region_stc.plot(subject="fsaverage", subjects_dir=fs_dir, surface="white", time_viewer=True)    
        region_plot.add_foci(region_stc.lh_vertno, coords_as_verts=True, hemi="lh", color="blue",scale_factor=0.2)
    except:
        print("No data in label")


#visualisering af data
# MEG
MEG_plot=MEGstc_morphed.plot(subject="fsaverage",subjects_dir=fs_dir,surface="white")
#add source locations as blue dots
#MEG_plot.add_foci(MEGstc_morphed.lh_vertno, coords_as_verts=True, hemi="lh", color="blue",scale_factor=0.2)
# man kan også lege med typen af surface
# fMRI
FMRI_plot=FMRIstc_morphed.plot(subjects_dir=fs_dir,surface="white",)
#add source locations as blue dots
#FMRI_plot.add_foci(FMRIstc_morphed.lh_vertno, coords_as_verts=True, hemi="lh", color="blue",scale_factor=0.2)

# Show the BEM surfaces used to construct the "forward model"
mne.viz.plot_bem(subject, fs_dir) # brain_surfaces="white"
