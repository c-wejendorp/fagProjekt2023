import numpy as np


def initializeVoxels(T, s, m, eeg_only = 2, meg_only = 2, fmri_only = 2, eeg_meg_shared = 1,
                     eeg_fmri_shared = 1, meg_fmri_shared = 1, golden_voxel = 1):
    """initializes however many voxels we want"""
    V = eeg_only + meg_only + fmri_only + eeg_meg_shared + meg_fmri_shared + eeg_fmri_shared + golden_voxel
    #initialize empty X matrix
    X = np.zeros((m, s, T, V))
    
    timestamps = np.array_split(list(range(T)), V)
    s = s - 1
    #generate signal for the modality-unique voxels
    for voxel in range(eeg_only):
        X[0, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
    for voxel in range(eeg_only, eeg_only + meg_only):
        X[1, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
    for voxel in range(eeg_only + meg_only, eeg_only + meg_only + fmri_only):
        X[2, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
    
    #generate pairwise-combined signal
    start_comb_idx = eeg_only + meg_only + fmri_only
    for voxel in range(start_comb_idx, start_comb_idx + eeg_meg_shared):
        X[0, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
        X[1, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
    for voxel in range(start_comb_idx + eeg_meg_shared, start_comb_idx + eeg_meg_shared + eeg_fmri_shared):
        X[0, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
        X[2, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
    for voxel in range(start_comb_idx + eeg_meg_shared + eeg_fmri_shared, start_comb_idx + eeg_meg_shared + eeg_fmri_shared + meg_fmri_shared):
        X[1, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
        X[2, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
           
    #generate the golden voxel all three modalities share
    golden_idx = start_comb_idx + eeg_meg_shared + eeg_fmri_shared + meg_fmri_shared
    for voxel in range(golden_idx, golden_idx + golden_voxel):
        X[0, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
        X[1, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
        X[2, s, timestamps[voxel], voxel] = np.random.normal(1, 0.1, size = len(timestamps[voxel]))
    
    return X

Xeeg = initializeVoxels(T = 100, s = 1, m = 3, eeg_only=1, meg_only=1, fmri_only=1, eeg_meg_shared=1, eeg_fmri_shared=0, meg_fmri_shared=0, golden_voxel=1)
#print(X)

