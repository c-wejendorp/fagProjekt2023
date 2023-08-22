import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
#from loadData import Real_Data

# define paths
path = "data/MMAA_results/multiple_runs/"
savepath = "MMAA/plots/"

modalityComb = ["eeg", "meg", "fmri"]
folder = path + f'{"-".join(modalityComb)}/'
savepath = savepath + f'{"-".join(modalityComb)}/'

color_dict = {"eeg": "blue", "meg": "red", "fmri": "green", "sum": "black"}

archetypRange = np.arange(2, 20+1, 2)    

# loop over splits
for split in [0,1]:
    # training loss, note this value has not been averaged over seeds
    datapath = folder + f"split_{split}/loss/"
    
    all_train_losses = []
    for modality in modalityComb:
        means_train = []
        stds_train = []
        
        # loop over all archetypes and seeds
        for k in archetypRange:
            loss_list_arrays = []
            
            for seed in range(0,91,10):
                loss_pr_iterations = np.load(datapath + f"loss_split-{split}_k-{k}_seed-{seed}_type-{modality}.npy")
                loss_list_arrays.append(loss_pr_iterations)
                
            # find the mean loss pr iteration 
            loss_list_arrays = np.array(loss_list_arrays)
            loss_list = np.mean(loss_list_arrays, axis=0)   

            plt.plot(loss_list, label=f"{modality}_train_k={k}", color=color_dict[modality], linestyle="solid")  
        
            # plot it
        #plt.errorbar(archetypRange,means_train,yerr=stds_train,label=f"{modality}_train",color=color_dict[modality],linestyle="solid")
        # add to all train losses
        #all_train_losses.append(means_train)     
    plt.legend()   
    plt.show()
    #plt.legend()
    #plot summarized train loss
    # sum over modalities
    all_train_losses = np.array(all_train_losses)
    all_train_losses = np.sum(all_train_losses,axis=0)

    plt.plot(archetypRange, all_train_losses, label="sum_train", color=color_dict["sum"], linestyle="solid")

