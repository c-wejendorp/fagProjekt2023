import numpy as np
#This is work in progress.
# currently i'm just testing stuff regarding the HPC
#read in the information from the models
C = np.load("MMAA/modelsInfo/C_matrix_k14_s0_split0.npy")
S = np.load("MMAA/modelsInfo/S_matrix_k14_s0_split0.npy")
eeg_loss = np.load("MMAA/modelsInfo/eeg_loss14_s0_split0.npy")
meg_loss = np.load("MMAA/modelsInfo/meg_loss14_s0_split0.npy")
fmri_loss = np.load("MMAA/modelsInfo/fmri_loss14_s0_split0.npy")
loss_adam = np.load("MMAA/modelsInfo/loss_adam14_s0_split0.npy")

#plot the loss
import matplotlib.pyplot as plt
plt.plot(eeg_loss, label = "EEG")
plt.plot(meg_loss, label = "MEG")
plt.plot(fmri_loss, label = "fMRI")
plt.plot(loss_adam, label = "Adam")
plt.legend()
plt.show()
