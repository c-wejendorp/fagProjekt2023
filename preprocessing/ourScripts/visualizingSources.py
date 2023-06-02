from pathlib import Path
import mne

#path to the freesurfer directory
#fs_dir = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition/freesurfer")
fs_dir = Path("data/freesurfer")

#path to the data directory (JespersProcessed)
#data_dir = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition_dtu/data")
data_dir = Path("data/JesperProcessed")
subject = "sub-10"

subject_dir = data_dir / subject
meg_dir = subject_dir / "ses-meg"
fwd_dir = meg_dir / "stage-forward"
pre_dir = meg_dir / "stage-preprocess"
inv_dir = meg_dir / "stage-inverse"

mri_dir = subject_dir / "ses-mri"
fmri_dir = mri_dir / "func"


"""

stc = mne.read_source_estimate(inv_dir / "task-facerecognition_cond-famous_fwd-mne_ch-eeg_split-0_stc")
stc_fs = mne.compute_source_morph(stc, subject, 'fsaverage', subjects_dir=fs_dir,
                                  smooth=5, verbose='error').apply(stc)

brain = stc_fs.plot(subjects_dir=fs_dir, initial_time=0.1,
                    clim=dict(kind='value', lims=[3, 6, 9]),
                    hemi='both', size=(1000, 500),
                    smoothing_steps=5, time_viewer=False,
                    add_data_kwargs=dict(
                        colorbar_kwargs=dict(label_font_size=10)))

brain=stc_fs.plot('fsaverage', subjects_dir=fs_dir)


morph = mne.read_source_morph(fwd_dir / "task-facerecognition_fwd-mne_morph.h5")
morphed=morph.apply(stc)

initial_time = 0.1

brain=morphed.plot('fsaverage', subjects_dir=fs_dir,initial_time=initial_time)
#brain=morphed.plot(subject, subjects_dir=fs_dir,initial_time=initial_time)
#idxs=MEGstc_morphed.lh_vertno
brain.add_foci(morphed.lh_vertno, coords_as_verts=True, hemi="lh", color="blue",scale_factor=0.2)
"""


initial_time = 0.1


MEGstc_morphed = mne.read_source_estimate(inv_dir / "task-facerecognition_space-fsaverage_cond-famous_fwd-mne_ch-eeg_split-0_stc")

#peak_vertex, peak_time = MEGstc_morphed.get_peak(hemi="lh", vert_as_index=True, time_as_index=True)

#peak_vertex_surf = MEGstc_morphed.lh_vertno[peak_vertex]

#peak_value = MEGstc_morphed.lh_data[peak_vertex, peak_time]
#add_foci(peak_vertex_surf, coords_as_verts=True, hemi="lh", color="blue")

brain=MEGstc_morphed.plot(subject, subjects_dir=fs_dir,initial_time=initial_time)
#idxs=MEGstc_morphed.lh_vertno
brain.add_foci(MEGstc_morphed.lh_vertno, coords_as_verts=True, hemi="lh", color="blue",scale_factor=0.2)

# Extract the fsaverage surface
#fsaverage_surface = MEGstc_morphed._morph(subject_from=src_subject, subject_to='fsaverage')

# The fsaverage surface is available in 'fsaverage_surface'

# Plot the fsaverage surface
#mne.viz.plot_surface(fsaverage_surface, color='gray', opacity=1.0)
test=1