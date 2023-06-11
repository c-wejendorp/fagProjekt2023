import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from loadData import Real_Data
from nmi import nmi
#This is work in progress.
# currently i'm just testing stuff regarding the HPC
#read in the information from the models
datapath = "data/MMAA_results/multiple_runs/split_0/"

number_of_seeds = 2

# loop over all archetypes
for k in range(2,10,2):
    # load the average S matrix for each seed
    S_matrices=[]
    for s in range(number_of_seeds):
        seed = s*10
        S_matrices.append(np.load(datapath + f"S_matrix_k{k}_s{seed}_split0.npy"))
    
    # calculate the NMI for S1 and S2, S2 and S3, etc and last S10 and S1
    # this needs to be done for each modality
    eeg_NMIs = []
    meg_NMIs = []
    fmri_NMIs = []
    for s in range(number_of_seeds):
        eeg_NMIs.append(nmi(S_matrices[s][0], S_matrices[(s+1)%number_of_seeds][0]))
        meg_NMIs.append(nmi(S_matrices[s][1], S_matrices[(s+1)%number_of_seeds][1]))
        fmri_NMIs.append(nmi(S_matrices[s][2], S_matrices[(s+1)%number_of_seeds][2]))
    # calculate the average NMI for each modality
    eeg_NMI_mean = np.mean(eeg_NMIs)
    meg_NMI_mean = np.mean(meg_NMIs)
    fmri_NMI_mean = np.mean(fmri_NMIs)
    # calculate the standard deviation for each modality
    eeg_NMI_std = np.std(eeg_NMIs)
    meg_NMI_std = np.std(meg_NMIs)
    fmri_NMI_std = np.std(fmri_NMIs)
    # plot the average NMI for each modality
    plt.errorbar(k, eeg_NMI_mean, yerr=eeg_NMI_std, fmt='o', color='green')
    plt.errorbar(k, meg_NMI_mean, yerr=meg_NMI_std, fmt='o', color='red')
    plt.errorbar(k, fmri_NMI_mean, yerr=fmri_NMI_std, fmt='o', color='blue')

plt.legend(["EEG", "MEG", "fMRI"])
plt.xlabel("Number of archetypes")
plt.ylabel("NMI")
plt.savefig("NMI.png")
plt.show()





#plot the different archetypes
"""
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
plt.savefig("test.png")
#plt.show()
"""

