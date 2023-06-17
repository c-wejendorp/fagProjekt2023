import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
from loadData import Real_Data
#This is work in progress.
# currently i'm just testing stuff regarding the HPC
#read in the information from the models
datapath = "MMAA/modelsInfo/"



#C = np.load(datapath + "C_matrix_k14_s0_split0.npy")
C = np.load("data\MMAA_results\multiple_runs\eeg-meg-fmri\split_0\C\C_split-0_k-40_seed-0.npy")
#S = np.load(datapath + "S_matrix_k14_s0_split0.npy")
# S = np.load("data\MMAA_results\multiple_runs\eeg-meg-fmri\split_0\S\S_split-0_k-14_seed-0_sub-avg.npy")

#plot the different archetypes
split = 0
k=14
X = Real_Data(subjects=range(1, 17), split=split)
X = [X.EEG_data, X.MEG_data, X.fMRI_data]
T = np.array([X[0].shape[1], X[1].shape[1], X[2].shape[1]]) #number of time points
V = X[0].shape[2] #number of sources


#plot archetypes
_, ax = plt.subplots(4)     

#plot the different archetypes
for m in range(3):
    A = np.mean(X[m]@C, axis = 0)    
    for arch in range(k):
        ax[m].plot(range(T[m]), A[:, arch])
ax[-1].plot(range(V), C)
plt.savefig("test2.png")
#plt.show()