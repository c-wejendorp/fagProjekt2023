import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
from loadData_oldway import Real_Data_oldway
from tqdm import tqdm


datapath = '/work3/s204090/data/MMAA_results/multiple_runs/time_conc/eeg-meg-fmri/split_0'
# we have decided to use 16 archetypes
k = 26
Cs = []
#S = []
for seed in range(0,91,10):
    C = np.load(datapath + f"/C/C_split-0_k-{k}_seed-{seed}.npy")
    Cs.append(C)
    #S_avg= np.load(datapath + f"/S/S_split-0_k-16_seed-{seed}_sub-avg.npy")    
    #S.append(S_avg)

#average over seeds
C = np.mean(Cs, axis=0)
#S = np.mean(S, axis=0)

#plot the different archetypes
split = 0
X = Real_Data_oldway(subjects=range(1, 17), split=split)
X = [X.EEG_data, X.MEG_data, X.fMRI_data]
T = np.array([X[0].shape[1], X[1].shape[1], X[2].shape[1]]) #number of time points
V = X[0].shape[2] #number of sources


#plot archetypes
_, ax = plt.subplots(3)     

#create dir to save plots
plotpath = 'MMAA/archeTypePlots'
if not os.path.exists(plotpath):
    os.makedirs(plotpath)

# plot the archetypes 
#plot the different archetypes

A_dict = {"A0": np.mean(X[0]@C, axis = 0),"A1": np.mean(X[1]@C, axis = 0),"A2": np.mean(X[2]@C, axis = 0)}

for arch in tqdm(range(k)):
    #plot archetypes
    _, ax = plt.subplots(3) 
    #plot the different archetypes
    ax[0].set_title(f'Archetype {arch+1} for EEG')
    ax[1].set_title(f'Archetype {arch+1} for MEG')
    ax[2].set_title(f'Archetype {arch+1} for fMRI') 
    # add horizontal line after each 180 time points
    for i in range(3):
        for j in range(1,3):
            ax[i].axvline(x=180*j, color='grey', linestyle='--') 

    for m in tqdm(range(3)):
        ax[m].plot(range(T[m]), A_dict[f"A{m}"][:, arch],color='black')   
    # save for each archetype
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(plotpath + f"/archetype_{arch+1}.png",dpi=300)
    # clear plot
    plt.clf()
    plt.close()

# close and destroy previous plot
plt.clf()
plt.close()

#plot all archetypes
_, ax = plt.subplots(3)  
ax[0].set_title(f'All {k} Archetypes for EEG')
ax[1].set_title(f'All {k} Archetypes for MEG')
ax[2].set_title(f'All {k} Archetypes for fMRI')
for i in range(3):
    for j in range(1,3):
        ax[i].axvline(x=180*j, color='grey', linestyle='--') 

for m in tqdm(range(3)):
    A = np.mean(X[m]@C, axis = 0)    
    for arch in tqdm(range(k)):
        ax[m].plot(range(T[m]), A[:, arch])
        # save for eahc archetype
        #plt.savefig(plotpath + f"/archetype_{arch}_modality_{m}.png",dpi=300)

# add space between plots
plt.subplots_adjust(hspace=0.5)
# add title and axis labelsplt
#ax[-1].plot(range(V), C)
plt.savefig(plotpath + "allArcheTypes.png",dpi=300)
#plt.show()