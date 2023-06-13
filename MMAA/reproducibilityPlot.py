import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from loadData import Real_Data
from nmi import nmi

# for split 0 we want to plot the average NMI with error bars for each modality for each number of archetypes.
# In the same plot we will also show the best average NMI across splits for EEG and MEG. (fmri is shared in both splits) 

number_of_seeds = 10
mods = ["eeg", "meg", "fmri"]
colors = ["green", "red", "blue"]

path = f"data/MMAA_results/multiple_runs/{'-'.join(mods)}/"

#here define which split is test and which is train
train = 0
test = 1

# lets start with split train

datapath = f"data/MMAA_results/multiple_runs/{'-'.join(mods)}/split_{train}/NMI/"

# loop over modalities
offset = 0
for idx, modality in enumerate(mods):
    NMI_tuples = []
    #now over archetypes
    
    for k in range(2,20+1,2):
        NMI_tuples.append(np.load(datapath + f"NMI_split-{train}_k-{k}_type-{modality}.npy"))
    
    # now we have a list of tuples with (mean,std) for each k
    # we want to plot the mean with error bars
    NMI_mean = [t[0] for t in NMI_tuples]
    NMI_std = [t[1] for t in NMI_tuples]
    plt.errorbar(np.arange(2,20+1,2)+offset, NMI_mean, yerr=NMI_std, fmt='o', capsize=5,color=colors[idx], label=f"{modality}_split_{train}")
    offset += 0.3

# now for the best NMI across splits for EEG and MEG

path = f"data/MMAA_results/multiple_runs/"

offset = 0
for idx, modality in enumerate(["eeg", "meg"]):
    NMI_best = []
    #now over archetypes
    for k in range(2,20+1,2):        
        NMI_best.append(max([np.load(f"data/MMAA_results/multiple_runs/{'-'.join(mods)}/split_{split}/NMI/NMI_split-{split}_k-{k}_type-{modality}.npy")[2] for split in [train, test]]))
        #NMI_best.append(max([np.load(path + f"split_{split}/NMI/NMI_split-{split}_k-{k}_type-{modality}.npy")[2] for split in [train, test]]))


    # plot it as a dotted line
    plt.plot(np.arange(2,20+1,2)+offset, NMI_best, '--', color=colors[idx], label=f"{modality}_best btw splits")
    
    offset += 0.3   

plt.xticks(np.arange(2,20+1,2))    

plt.legend()
plt.xlabel("Number of archetypes")
plt.ylabel("NMI")
plt.savefig(f"MMAA/plots/NMI_split={train}.png", dpi=300)
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

