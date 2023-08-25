from pathlib import Path
import mne
import numpy as np

# this file differs from plot_mod_on_brain by plotting the conditions 
# each brain plot is therefore a modality, a condition and an archetype

# path to the freesurfer directory
fs_dir = Path("data/freesurfer")
subject = "sub-01"

# read labels for corpus callosum
label_lh = mne.read_label(fs_dir / "fsaverage/label/lh.Medial_wall.label",subject=subject)
label_rh = mne.read_label(fs_dir / "fsaverage/label/rh.Medial_wall.label",subject=subject)
label_both = np.concatenate((label_lh.vertices, label_rh.vertices + 10242))

# remove corpus callosum vertices from total number of vertices
dipoles = np.arange(20484) 
dipoles = np.delete(dipoles, label_both) 

def plot_c_on_brain(path, split, seed, k, fs_dir = Path("data/freesurfer")):
    """function that displays (no_of_conditions * no_of_archetypes) plots
    of the found source coefficients defining the k'th archetype (columns of C)
    as heat maps on the brain surface.
    
    path (str): path to the analysis with the data from split 0 and 1
    split (int): whether you want to evaluate data from split 0 or 1
    seed (list of int): which seed value(s) to load data from
    k (int): the number of archetypes used in the analysis
    fs_dir (Path obj): used for plotting as the subject_dir arg"""
    
    # load data
    split_path = path + f"/split_{split}"
    matrix_path = split_path + f"/C"
    
    # load all seed matrices for a given split and number of archetypes
    data = []
    for s in seed:
        data.append(np.load(matrix_path + f"/C_split-{split}_k-{k}_seed-{s}.npy"))
        
    # average c matrix across all seeds
    data = np.mean(np.asarray(data), axis = 0)
    
    # define conditions by splitting the data in equal thirds
    cond = [data[:data.shape[0] // 3, :], data[data.shape[0] // 3:data.shape[0] // 3 * 2], data[data.shape[0] // 3 * 2: ]]

    # which hemisphere(s) to plot on. "lr", "rh" and "both" are allowed
    hemi = ["both"]
    
    # make a plot for all conditions and archetype combinations on the brain hemisphere(s)
    for condition in cond:
        for h in hemi:
            for archetypes in range(k):
                arch = condition[:, archetypes]
                
                # load data as a source estimate object and plot
                overlay = mne.SourceEstimate(arch, vertices = [dipoles[:9354], dipoles[9354:] - 10242], tmin = 0, tstep = 1)
                
                # plot on the hemisphere(s) and add most activated source as foci
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

def plot_s_on_brain(path, split, seed, k, fs_dir = Path("data/freesurfer"), mean = False, std = False, subject = None):
    """function that displays (no_of_conditions * no_of_archetypes) plots
    of sources using the k'th archetype to reconstruct (rows of S)
    as heat maps on the brain surface.

    Args:
        path (str): path to analysis data containing data from split 0 and split 1
        split (int): 0 or 1 allowed. defines which data split you want to plot
        seed (list of int): all seed values used in the analysis
        k (int): the number of archetypes used in the analysis
        fs_dir (Path obj, optional): used for plotting when defining the 
        subject's directory. Defaults to Path("data/freesurfer").
        mean (bool, optional): plots the mean source value across all subjects. Defaults to False.
        std (bool, optional): plots the std of each source across all subjects. Defaults to False.
        subject (int, optional): plot a single subject. Defaults to None.

    Raises:
        Exception: either mean, std or subject has to be True/not None. however, mean and std cannot both be True
                   and subject can only be None if either mean or std is True and vice versa
    """
    
    # load data
    split_path = path + f"/split_{split}"
    matrix_path = split_path + f"/Sms"
    
    # load all seed matrices
    data = []
    for s in seed:
        data.append(np.load(matrix_path + f"/Sms_split-{split}_k-{k}_seed-{s}.npy"))
        
    # average s matrix across all seeds
    data = np.mean(np.asarray(data), axis = 0)
    
    # error handling
    if (mean and std) or ((mean or std) ^ (subject is None)):
        raise Exception("one of the problems occured: \n" +
                        "1. mean and std cannot both be True \n" +
                        "2. you want data from a single subject while having set mean and/or std = True \n"
                        "3. you have chosen to neither use the mean, the standard deviation " + 
                        "nor have you chosen data from a single subject. the code " +
                        "needs one of the three to be fulfilled")
    elif mean:
        # average across subjects
        data = np.mean(data, axis = 1)
    elif std:
        # std across subjects
        data = np.std(data, axis = 1)
        
    if subject is not None:
        # plot a single subject
        data = data[:, subject, :, :]
    
    # loop over all modalities
    for m in range(data.shape[0]):
        data_mod = data[m].T
        
        # define conditions by splitting the data in equal thirds
        cond = [data_mod[:data_mod.shape[0] // 3, :], data_mod[data_mod.shape[0] // 3:data_mod.shape[0] // 3 * 2, :], data_mod[data_mod.shape[0] // 3 * 2:, :]]
        
        # which hemisphere(s) to plot on. "lr", "rh" and "both" are allowed
        hemi = ["both"]
        
        # make plot for all conditon and archetype combinations on the brain surface
        for condition in cond:
            for h in hemi:
                for archetypes in range(k):
                    arch = condition[:, archetypes]
                    
                    # load data as a source estimate object and plot
                    overlay = mne.SourceEstimate(arch, vertices = [dipoles[:9354], dipoles[9354:] - 10242], tmin = 0, tstep = 1)
                    
                    # plot on the hemisphere(s) and add most activated source as foci
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

    c = False
    s = True
    mean = False
    std = False
    subj = None
    
    if c:
        # plot c matrix on the brain
        plot_c_on_brain(trimodal_path, split = split, k = k, seed = range(0, 100, 10))
    if s:
        # plot s matrix on the brain
        plot_s_on_brain(trimodal_path, split = split, k = k, seed = range(0, 100, 10), mean = mean, std = std, subject = subj)