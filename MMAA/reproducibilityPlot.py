import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from loadData import Real_Data
#This is work in progress.
# currently i'm just testing stuff regarding the HPC
#read in the information from the models
datapath = "MMAA/modelsInfo/"

C = np.load(datapath + "C_matrix_k14_s0_split0.npy")
S = np.load(datapath + "S_matrix_k14_s0_split0.npy")
eeg_loss = np.load(datapath + "eeg_loss14_s0_split0.npy")
meg_loss = np.load(datapath + "meg_loss14_s0_split0.npy")
fmri_loss = np.load(datapath + "fmri_loss14_s0_split0.npy")
loss_adam = np.load(datapath + "loss_adam14_s0_split0.npy")

#plot th loss curves in seperate plots
plt.plot(eeg_loss)
plt.title("EEG loss")
plt.savefig("eeg_loss.png")
plt.show()

plt.plot(meg_loss)
plt.title("MEG loss")
plt.savefig("meg_loss.png")
plt.show()

plt.plot(fmri_loss)
plt.title("fMRI loss")
plt.savefig("fmri_loss.png")
plt.show()

plt.plot(loss_adam)
plt.title("Adam loss")
plt.savefig("adam_loss.png")
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

