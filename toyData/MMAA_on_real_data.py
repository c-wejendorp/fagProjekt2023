from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np
from toyDataAAMulti import MMAA
import torch 

#path to the freesurfer directory
#fs_dir = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition/freesurfer")
fs_dir = Path("data/freesurfer")

#path to the data directory (JespersProcessed)
#data_dir = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition_dtu/data")
data_dir = Path("data/JesperProcessed")

#NOTE this is not yet done in source space but just on channel level
# it is also only done on the modularites EEG and MEG (gradiometer)
# furthermore it is only done on two subjects (1 and 2)
# finally we only look at 1 channel () for in each modality for now

#let us load the data such that we have nested lists with subject, modality data


evos = []
numSubjects = 2
for i in range (1,numSubjects+1):   
    subject = f"sub-0{i}"
    subject_dir = data_dir / subject
    meg_dir = subject_dir / "ses-meg"
    fwd_dir = meg_dir / "stage-forward"
    pre_dir = meg_dir / "stage-preprocess"
    #inv_dir = meg_dir / "stage-inverse"
    #mri_dir = subject_dir / "ses-mri"
    #fmri_dir = mri_dir / "func"
# ERP
    evo = mne.read_evokeds(pre_dir / "task-facerecognition_proc-p_cond-famous_split-0_evo.fif")
    evo = evo[0]
    evos.append(evo)
# lets have the structure of modality, subject, channeldata
numModalities = 2
numSubjects = 2
#create inital nested list
nestedList = [[0]*numSubjects for modality in range(numModalities)]
#fill the nested list
for idx_modality, modality in enumerate(["eeg", "grad"]):
    for idx_subject, evo in enumerate(evos):
        # we transpose the data such that we have time x channels

        #nestedList[idx_modality][idx_subject] = evo.get_data(modality).T

        # we also just picks the first two channel for now
        nestedList[idx_modality][idx_subject] = evo.get_data(modality)[:2].T
        

#plot the data in one plot and two figures

for idx_modality, modality in enumerate(["eeg", "grad"]):
    for idx_subject, evo in enumerate(evos):
        plt.plot(nestedList[idx_modality][idx_subject], label=f"subject {idx_subject+1}")
    plt.title(modality)
    plt.legend()
    plt.show()


#turn the nested list into a numpy array , can be done smarter

T,V = nestedList[0][0].shape
Xms = np.zeros((numModalities, numSubjects, T, V))    
for i in range(numModalities):
    for j in range(numSubjects):
        Xms[i,j,:,:] = nestedList[i][j]


# now we try to do the MMAA on the data
def realDataMMA(Xms,numArchetypes=10,numpySeed=32,torchSeed=0,plotDistributions=False,learningRate=1e-3,numIterations=10000):

    np.random.seed(numpySeed)
    torch.manual_seed(torchSeed)

    ### num modulaties, num subjects, time steps, num voxel

    numModalities = Xms.shape[0]
    numSubjects = Xms.shape[1]
    T = Xms.shape[2]
    V = Xms.shape[3]

    k = numArchetypes

    #convert to torch tensor
    Xms = torch.tensor(Xms)

    #hyperparameters
    lr = learningRate
    n_iter = numIterations

    #loss function
    model = MMAA(V, T, k, Xms, numModalities=numModalities, numSubjects=numSubjects)
    lossCriterion = torch.nn.MSELoss(reduction = "sum")
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Creating Dataloader object
    loss_Adam = []
    for i in range(n_iter):
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # making a prediction in forward pass
        Xrecon = model.forward()
        # calculating the loss between original and predicted data points
        loss = lossCriterion(Xrecon, Xms)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()
        # store loss into list
        loss_Adam.append(loss.item())

    #print("loss list ", loss_Adam) 
    print("final loss: ", loss_Adam[-1])
    Xrecon = model.XCSms

    Xrecon = model.XCSms.detach().numpy()
    for idx_modality, modality in enumerate(["eeg", "grad"]):
        for idx_subject, evo in enumerate(evos):
            plt.plot(Xrecon[idx_modality,idx_subject,:,:], label=f"subject {idx_subject+1}")
            plt.title(f"The reconstructed data, modality {modality}, subject {idx_subject+1}")
            plt.legend()
            plt.show()   


            
    
    #return data,archeTypes,loss_Adam

if __name__ == "__main__":
    realDataMMA(Xms,numIterations=2000,numArchetypes=180)






       
