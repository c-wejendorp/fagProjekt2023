import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
from loadData_oldway import Real_Data_oldway
from tqdm import tqdm


datapath = '/work3/s204090/data/MMAA_results/multiple_runs/time_conc/eeg-meg-fmri/split_0'
# we have decided to use 16 archetypes

Cs = []
#S = []
for seed in range(0,91,10):
    C = np.load(datapath + f"/C/C_split-0_k-16_seed-{seed}.npy")
    Cs.append(C)
    #S_avg= np.load(datapath + f"/S/S_split-0_k-16_seed-{seed}_sub-avg.npy")    
    #S.append(S_avg)

#average over seeds
C = np.mean(Cs, axis=0)
#S = np.mean(S, axis=0)

#plot the different archetypes
split = 0
k=16 
X = Real_Data_oldway(subjects=range(1, 17), split=split)
X = [X.EEG_data, X.MEG_data, X.fMRI_data]
T = np.array([X[0].shape[1], X[1].shape[1], X[2].shape[1]]) #number of time points
V = X[0].shape[2] #number of sources


#plot archetypes
_, ax = plt.subplots(3)     

#plot the different archetypes
for m in tqdm(range(3)):
    A = np.mean(X[m]@C, axis = 0)    
    for arch in tqdm(range(k)):
        ax[m].plot(range(T[m]), A[:, arch])

# add space between plots
plt.subplots_adjust(hspace=0.5)
# add title and axis labels
ax[0].set_title('Archetypes for EEG')
ax[1].set_title('Archetypes for MEG')
ax[2].set_title('Archetypes for fMRI')
#ax[-1].plot(range(V), C)
plt.savefig("testArcheTypes.png",dpi=300)
#plt.show()