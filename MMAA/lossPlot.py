import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
from loadData import Real_Data
#This is work in progress.
# currently i'm just testing stuff regarding the HPC
#read in the information from the models

#datapath = "MMAA/data/MMAA_results/multiple_runs/split_0/"

def readLossFromFile(datapath,file,array = False):
    numArch = int(file.split("_")[1][4:])   
    if array:
        loss = np.load(datapath + file)   
    else :
        loss = np.load(datapath + file)[-1]

    modelSeed = int(file.split("_")[2][1:])

    return [numArch,loss,modelSeed]

def createLossPlot1(datapath = "data/MMAA_results/multiple_runs/split_0/",savepath = "MMAA/plots/"):
    #open all files starting with eeg
    eeg_loss = []
    meg_loss = []
    fmri_loss = []
    adam_loss = []
    for file in os.listdir(datapath):
        if file.startswith("eeg"):
            eeg_loss.append(readLossFromFile(datapath,file))            
        elif file.startswith("meg"):
            meg_loss.append(readLossFromFile(datapath,file))
        elif file.startswith("fmri"):
            fmri_loss.append(readLossFromFile(datapath,file))
        elif file.startswith("loss_adam"):
            adam_loss.append(readLossFromFile(datapath,file))

    #sort the lists
    eeg_loss.sort(key = lambda x: x[0])
    meg_loss.sort(key = lambda x: x[0])
    fmri_loss.sort(key = lambda x: x[0])
    adam_loss.sort(key = lambda x: x[0])

    # calculate the mean and std for each number of archetypes
    eeg_mean = np.mean([x[1] for x in eeg_loss])
    eeg_std = np.std([x[1] for x in eeg_loss])
    meg_mean = np.mean([x[1] for x in meg_loss])
    meg_std = np.std([x[1] for x in meg_loss])
    fmri_mean = np.mean([x[1] for x in fmri_loss])
    fmri_std = np.std([x[1] for x in fmri_loss])
    adam_mean = np.mean([x[1] for x in adam_loss])
    adam_std = np.std([x[1] for x in adam_loss])
    
    #plot the mean values with std
    plt.errorbar([x[0] for x in eeg_loss], [x[1] for x in eeg_loss], yerr = eeg_std, label = "eeg")
    plt.errorbar([x[0] for x in meg_loss], [x[1] for x in meg_loss], yerr = meg_std, label = "meg")
    plt.errorbar([x[0] for x in fmri_loss], [x[1] for x in fmri_loss], yerr = fmri_std, label = "fmri")
    plt.errorbar([x[0] for x in adam_loss], [x[1] for x in adam_loss], yerr = adam_std, label = "adam")
    plt.legend()
    # make the x ticks integers 
    plt.xticks([x[0] for x in eeg_loss])
    plt.title("Final loss for different number of archetypes training data")
    plt.xlabel("Number of archetypes")
    plt.ylabel("Final loss")
    plt.savefig(savepath + "finalLoss.png")    
    #plt.show()    

def createLossPlot2(datapath = "data/MMAA_results/multiple_runs/split_0/",savepath = "MMAA/plots/"):
    #open all files starting with eeg
    eeg_loss = []
    meg_loss = []
    fmri_loss = []
    adam_loss = []
    for file in os.listdir(datapath):
        if file.startswith("eeg"):
            eeg_loss.append(readLossFromFile(datapath,file,array = True))            
        elif file.startswith("meg"):
            meg_loss.append(readLossFromFile(datapath,file,array = True))
        elif file.startswith("fmri"):
            fmri_loss.append(readLossFromFile(datapath,file,array=True))
        elif file.startswith("loss_adam"):
            adam_loss.append(readLossFromFile(datapath,file,array=True))
    
    #sort the lists
    eeg_loss.sort(key = lambda x: x[0])
    meg_loss.sort(key = lambda x: x[0])
    fmri_loss.sort(key = lambda x: x[0])
    adam_loss.sort(key = lambda x: x[0])

    # for each number of archetypes, calculate the mean and std for each seed
    eeg_mean = []    
    meg_mean = []
    fmri_mean = []
    adam_mean = []
    
    for i in range(2,20+1,2):
        # the loss arrays are of different length
        # we we use the lenght of the shortest array
        # this is not the best solution, but it works
        #eeg
        eeg_losses=[x[1] for x in eeg_loss if x[0] == i]
        eeg_its=min([len(x) for x in eeg_losses])
        eeg_mean.append(np.mean([x[:eeg_its] for x in eeg_losses],axis = 0))

        #meg
        meg_losses=[x[1] for x in meg_loss if x[0] == i]
        meg_its=min([len(x) for x in meg_losses])
        meg_mean.append(np.mean([x[:meg_its] for x in meg_losses],axis = 0))

        #fmri
        fmri_losses=[x[1] for x in fmri_loss if x[0] == i]
        fmri_its=min([len(x) for x in fmri_losses])
        fmri_mean.append(np.mean([x[:fmri_its] for x in fmri_losses],axis = 0))

        #adam
        adam_losses=[x[1] for x in adam_loss if x[0] == i]
        adam_its=min([len(x) for x in adam_losses])
        adam_mean.append(np.mean([x[:adam_its] for x in adam_losses],axis = 0))
        
    
    #plot the loss curve for each number of archetypes
    for i in range(len(eeg_mean)):
    #for i in range(len(eeg_mean)):
        #if i > 8:
        plt.plot(range(len(eeg_mean[i])),eeg_mean[i],label = f"eeg_k={2*(i+1)}")
        plt.plot(range(len(meg_mean[i])),meg_mean[i],label = f"meg_k={2*(i+1)}")
        plt.plot(range(len(fmri_mean[i])),fmri_mean[i],label = f"fmri_k={2*(i+1)}")
        plt.plot(range(len(adam_mean[i])),adam_mean[i],label = f"adam_k={2*(i+1)}")
        

   
    plt.legend()
    # make the x ticks integers
    plt.title("Mean Loss curve for different number of archetypes")
    plt.xlabel("Iteration")
    plt.ylabel("Final loss")
    plt.savefig(savepath + "lossCurve.png")
    #plt.show()


    


   
   
if __name__ == "__main__":
    createLossPlot1()
    #close all plots
    plt.close("all")
    createLossPlot2()