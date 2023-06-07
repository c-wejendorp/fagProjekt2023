from pathlib import Path
import mne
import numpy as np

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

#read labels for corpus callosum
label_lh = mne.read_label(fs_dir / "fsaverage/label/lh.Medial_wall.label",subject=subject)
label_rh = mne.read_label(fs_dir / "fsaverage/label/rh.Medial_wall.label",subject=subject)
label_both = np.concatenate((label_lh.vertices, label_rh.vertices + 10242))
    
#remove indices from the M/EEG corresponding to corpus callosum
MEGstc_morphed.vertices[0] = np.delete(MEGstc_morphed.vertices[0], label_lh.vertices)
MEGstc_morphed.vertices[1] = np.delete(MEGstc_morphed.vertices[1], label_rh.vertices)
MEGstc_morphed.data = np.delete(MEGstc_morphed.data, label_both, axis = 0)

#loading the c-matrix
c = np.load("MMAA/C_matrix.npy")

plot_per_arch = True

#thresholding the sources. currently thresholding for 0.05
thresh = 5e-2
which_sources = np.where(np.any(c >= thresh, axis = 1))[0]

#splitting all sources into left and right hemispheres
#we divide by 9354 because we know that the c-matrix is 18715 aka after removing
#corpus callosum. it has been previously noted that the shape af removal is [9354, 9361]
which_sources_lh = which_sources[which_sources < 9354]
which_sources_rh = which_sources[which_sources >= 9354] - 9354 #we subtract 9354 here because the vertex field starts from index 0

#copy the MEG-morphed object to newly index the activating sources
brain_plot = MEGstc_morphed.copy()

#an example:
#all vertices together: [0, 1, 2, 3, 4, 5]
#after deleting redundant regions: [0, 2, 3]
#the c matrix only gives us indices from the "after deleting region" i. e C = [0, 1]
#we therefore need to mask like so: after_delete[C] for certain threshold values for C
brain_plot.vertices[0] = brain_plot.vertices[0][which_sources_lh]
brain_plot.vertices[1] = brain_plot.vertices[1][which_sources_rh]
brain_plot.data = brain_plot.data[which_sources]

# #plot the significant sources from the c-matrix (after removing)
# c_matrix_plot = brain_plot.plot(subject="fsaverage", subjects_dir=fs_dir, surface="white", time_viewer=True)    
# c_matrix_plot.add_foci(brain_plot.lh_vertno, coords_as_verts=True, hemi="lh", color="blue",scale_factor=0.2)

if plot_per_arch:
    #thresholding the sources for each archetype
    sources_per_arch = [np.where(c[:, k] >= thresh)[0] for k in range(c.shape[1])]

    #splitting all sources into left and right hemispheres for each archetype
    arch_sources_lh = list(sources_per_arch[k][sources_per_arch[k] < 9354] for k in range(c.shape[1]))
    arch_sources_rh = list(sources_per_arch[k][sources_per_arch[k] >= 9354] - 9354 for k in range(c.shape[1])) #we subtract 9354 here because the vertex field starts from index 0

    lh_sources = []
    rh_sources = []
    for k in range(c.shape[1]):
        #copy the MEG-morphed object to newly index the activating sources
        arch_plot = MEGstc_morphed.copy()
        
        #append all archetype indices to a list 
        lh_sources.append(arch_plot.vertices[0][arch_sources_lh[k]])
        rh_sources.append(arch_plot.vertices[1][arch_sources_rh[k]])
        
#plot the significant sources from the c-matrix (after removing)
c_matrix_plot = brain_plot.plot(subject="fsaverage", subjects_dir=fs_dir, surface="white", time_viewer=True) 

if plot_per_arch:
    #random HEX-code generator
    import random
    number_of_colors = c.shape[1]
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                for __ in range(number_of_colors)]

    #plot the sources with a color assigned to each archetype
    for k in range(c.shape[1]):  
        archetype = list(np.where(np.isin(brain_plot.lh_vertno, lh_sources[k]))[0])
        c_matrix_plot.add_foci(brain_plot.lh_vertno[archetype], coords_as_verts=True, hemi="lh", color=color[k],scale_factor=0.2)
else:
    #if you don't care about archetypes and just want all plotted, just run this line and comment the loop out
    c_matrix_plot.add_foci(brain_plot.lh_vertno, coords_as_verts=True, hemi="lh", color="blue",scale_factor=0.2)


#plot the sources (after removing)
region_plot = MEGstc_morphed.plot(subject="fsaverage", subjects_dir=fs_dir, surface="white", time_viewer=True)    
region_plot.add_foci(MEGstc_morphed.lh_vertno, coords_as_verts=True, hemi="lh", color="blue",scale_factor=0.2)

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
