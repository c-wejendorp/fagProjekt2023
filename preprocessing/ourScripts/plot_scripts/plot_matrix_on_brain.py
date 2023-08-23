from pathlib import Path
import mne
import numpy as np

# this file differs from plot_mod_on_brain by plotting the conditions 
# each brain plot is therefore a modality, a condition and an archetype

#path to the freesurfer directory
fs_dir = Path("data/freesurfer")
subject = "sub-01"

#read labels for corpus callosum
label_lh = mne.read_label(fs_dir / "fsaverage/label/lh.Medial_wall.label",subject=subject)
label_rh = mne.read_label(fs_dir / "fsaverage/label/rh.Medial_wall.label",subject=subject)
label_both = np.concatenate((label_lh.vertices, label_rh.vertices + 10242))

#remove corpus callosum vertices from total number of vertices
dipoles = np.arange(20484) 
dipoles = np.delete(dipoles, label_both) 

def plot_c_on_brain(path, split, seed, k, fs_dir = Path("data/freesurfer")):
    """function that displays (no_of_conditions * no_of_archetypes) plots
    of the chosen source coefficients defining the k'th archetype
    as heat maps on the brain hemisphere.
    
    path (str): path to the analysis with the data from split 0 and 1
    split (int): whether you want to evaluate data from split 0 or 1
    seed (list of int): which seed value(s) to load data from
    fs_dir (Path obj): uses"""
    
    #load data
    split_path = path + f"/split_{split}"
    matrix_path = split_path + f"/C"
    
    #load all seed matrices
    data = []

    for s in seed:
        data.append(np.load(matrix_path + f"/C_split-{split}_k-{k}_seed-{s}.npy"))
        
    #average c matrix across all seeds
    data = np.mean(np.asarray(data), axis = 0)
    cond = [data[:data.shape[0] // 3, :], data[data.shape[0] // 3:data.shape[0] // 3 * 2], data[data.shape[0] // 3 * 2: ]]

    hemi = ["both"]
    for condition in cond:
        for h in hemi:
            #make a plot for each archetype
            for archetypes in range(k):
                arch = condition[:, archetypes]
                
                #load data as a source estimate object and plot
                overlay = mne.SourceEstimate(arch, vertices = [dipoles[:9354], dipoles[9354:] - 10242], tmin = 0, tstep = 1)
                
                #plot for both hemispheres and add most activated source as foci
                if h == "lh":
                    overlay_plot = overlay.plot(subject="fsaverage", subjects_dir=fs_dir, 
                                                surface="white", time_viewer=True, views = 'auto', colormap = "cool", 
                                                hemi = h, title = f"c_hemi_{h}_k_{archetypes}/{k}", 
                                                add_data_kwargs = dict(fmin = 0, fmid = max(overlay.data[:9354]) * 0.7, fmax = max(overlay.data[:9354]), smoothing_steps = 0))  
                    #overlay_plot.add_foci(overlay.lh_vertno[list(overlay.data[:9354]).index(max(overlay.data[:9354]))], coords_as_verts=True, hemi=h, color="white", scale_factor=0.2)
                    print("Checkpoint! Add a breakpoint here and take a picture!")
                elif h == "both":
                    overlay_plot = overlay.plot(subject="fsaverage", subjects_dir=fs_dir, 
                                                surface="inflated", time_viewer=True, views = 'auto', colormap = "cool", 
                                                hemi = h, title = f"c_hemi_{h}_k_{archetypes}/{k}", 
                                                add_data_kwargs = dict(fmin = 0, fmid = max(data[:, archetypes]) * 0.7, fmax = max(data[:, archetypes]), smoothing_steps = 0))  
                    #overlay_plot.add_foci(overlay.lh_vertno[list(overlay.data[:9354]).index(max(overlay.data[:9354]))], coords_as_verts=True, hemi=h, color="white", scale_factor=0.2)
                    print("Checkpoint! Add a breakpoint here and take a picture!")
                else:
                    overlay_plot = overlay.plot(subject="fsaverage", subjects_dir=fs_dir, 
                                                surface="white", time_viewer=True, views = 'auto', colormap = "cool",
                                                hemi = h, title = f"c_hemi_{h}_k_{archetypes}/{k}",
                                                add_data_kwargs = dict(fmin = 0, fmid = max(overlay.data[9354:]) * 0.7, fmax = max(overlay.data[9354:]), smoothing_steps = 0))       
                    #overlay_plot.add_foci(overlay.rh_vertno[list(overlay.data[9354:]).index(max(overlay.data[9354:]))], coords_as_verts=True, hemi=h, color="white", scale_factor=0.2)
                    print("Checkpoint! Add a breakpoint here and take a picture!")

def plot_s_on_brain(path, split, seed, k, thresh = 10e-5, fs_dir = Path("data/freesurfer"), mean = False, std = False, subject = None):
    #load data
    split_path = path + f"/split_{split}"
    matrix_path = split_path + f"/Sms"
    
    #load all seed matrices
    data = []
    
    for s in seed:
        data.append(np.load(matrix_path + f"/Sms_split-{split}_k-{k}_seed-{s}.npy"))
        
    #average s matrix across all seeds
    data = np.mean(np.asarray(data), axis = 0)
    if mean:
        #average across subjects
        data = np.mean(data, axis = 1)
    elif std:
        #std across subjects
        data = np.std(data, axis = 1)
    elif subject is not None:
        data = data[:, subject, :, :]
    else:
        raise Exception("you have chosen to neither use the mean, the standard deviation " + 
                        "nor have you chosen data from a single subject. the code " +
                        "needs one of the three to be fulfilled")
    
    for m in range(data.shape[0]):
        data_mod = data[m].T
        
        cond = [data_mod[:data_mod.shape[0] // 3, :], data_mod[data_mod.shape[0] // 3:data_mod.shape[0] // 3 * 2, :], data_mod[data_mod.shape[0] // 3 * 2:, :]]

        hemi = ["both"]
        for condition in cond:
            for h in hemi:
                #make a plot for each archetype
                for archetypes in range(k):
                    arch = condition[:, archetypes]
                    
                    #load data as a source estimate object and plot
                    overlay = mne.SourceEstimate(arch, vertices = [dipoles[:9354], dipoles[9354:] - 10242], tmin = 0, tstep = 1)
                    
                    #plot for both hemispheres and add most activated source as foci
                    if h == "lh":
                        overlay_plot = overlay.plot(subject="fsaverage", subjects_dir=fs_dir, 
                                                    surface="white", time_viewer=True, views = 'auto', colormap = "cool",
                                                    hemi = h, title = ["eeg", "meg", "fmri"][m] + f"_hemi_{h}_k_{archetypes}/{k}", 
                                                    add_data_kwargs = dict(fmin = 0, fmid = max(overlay.data[:9354]) * 0.9, fmax = max(overlay.data[:9354]), smoothing_steps = 0))       
                        #overlay_plot.add_foci(overlay.lh_vertno[list(overlay.data[:9354]).index(max(overlay.data[:9354]))], coords_as_verts=True, hemi=h, color="white", scale_factor=0.2)
                        print("Checkpoint! Add a breakpoint here and take a picture!")
                    elif h == "both":
                        overlay_plot = overlay.plot(subject="fsaverage", subjects_dir=fs_dir, 
                                                    surface="white", time_viewer=True, views = 'auto', colormap = "cool",
                                                    hemi = h, title = f"c_hemi_{h}_k_{archetypes}/{k}", 
                                                    add_data_kwargs = dict(fmin = 0, fmid = max(data_mod[:, archetypes]) * 0.7, fmax = max(data_mod[:, archetypes]), smoothing_steps = 0)) 
                        #overlay_plot.add_foci(overlay.lh_vertno[list(overlay.data[:9354]).index(max(overlay.data[:9354]))], coords_as_verts=True, hemi=h, color="white", scale_factor=0.2)
                        print("Checkpoint! Add a breakpoint here and take a picture!")
                    else:
                        overlay_plot = overlay.plot(subject="fsaverage", subjects_dir=fs_dir, 
                                                    surface="white", time_viewer=True, views = 'auto', colormap = "cool",
                                                    hemi = h, title = ["eeg", "meg", "fmri"][m] + f"_hemi_{h}_k_{archetypes}/{k}", 
                                                    add_data_kwargs = dict(fmin = 0, fmid = max(overlay.data[9354:]) * 0.9, fmax = max(overlay.data[9354:]), smoothing_steps = 0))       
                        #overlay_plot.add_foci(overlay.rh_vertno[list(overlay.data[9354:]).index(max(overlay.data[9354:]))], coords_as_verts=True, hemi=h, color="white", scale_factor=0.2)
                        print("Checkpoint! Add a breakpoint here and take a picture!")

if __name__ == "__main__":
    overlay_path = "data/MMAA_results/multiple_runs_spat"
    trimodal_path = overlay_path + "/eeg-meg-fmri"       

    split = 0
    k = 4

    #plot c matric on the brain
    #plot_c_on_brain(trimodal_path, split = split, k = k, seed = range(0, 100, 10))
    
    #plot mean value for subjects on the brain
    #plot_s_on_brain(trimodal_path, split = split, k = k, seed = range(0, 100, 10), mean = True)
    #plot ªstd value for subjects on the brain
    plot_s_on_brain(trimodal_path, split = split, k = k, seed = range(0, 100, 10), std = True)
    
    # #plot s matrix for one subject
    # plot_s_on_brain(trimodal_path, split = split, k = k, seed = range(0, 100, 10), mean = True, subject = 0)
    # plot_s_on_brain(trimodal_path, split = split, k = k, seed = range(0, 100, 10), mean = True, subject = 1)