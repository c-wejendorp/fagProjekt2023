import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
#from loadData import Real_Data

def loss_pr_archetype_plot(path="data/MMAA_results/multiple_runs/",savepath="MMAA/plots/",modalityComb=["eeg","meg","fmri"]):
    folder = path + f'{"-".join(modalityComb)}/'
    savepath = savepath + f'{"-".join(modalityComb)}/'

    color_dict = {"eeg":"blue","meg":"red","fmri":"green","sum":"black"}

    archetypRange = np.arange(2,40+1,2)
    

    for split in [0,1]:

        """
        #lets start with trainloss, note this value has not been averaged over seeds
        datapath = folder + f"split_{split}/loss/"

        # for the train loss
        all_train_losses = []

        for modality in modalityComb:
            means_train=[]
            stds_train=[]
            #now over archetypes
            for k in archetypRange:
                loss_list = []
                for seed in range(0,91,10):
                    loss_pr_iterations=np.load(datapath + f"loss_split-{split}_k-{k}_seed-{seed}_type-{modality}.npy")
                    loss_list.append(loss_pr_iterations[-1])
                
                # now find the mean and std
                means_train.append(np.mean(loss_list))
                stds_train.append(np.std(loss_list))
                
                # plot it
            plt.errorbar(archetypRange,means_train,yerr=stds_train,label=f"{modality}_train",color=color_dict[modality],linestyle="solid")
            # add to all train losses
            all_train_losses.append(means_train)       

        #plot summarized train loss
        # sum over modalities
        all_train_losses = np.array(all_train_losses)
        all_train_losses = np.sum(all_train_losses,axis=0)

        plt.plot(archetypRange,all_train_losses,label="sum_train",color=color_dict["sum"],linestyle="solid")
        """

        # now for the test loss
        # update datapath
        datapath = folder + f"split_{split}/test_loss/"

        all_test_losses = []
        # loop over modalities except fmri
        for modality in modalityComb:
            if modality == "fmri":
                 continue
            means_test=[]
            stds_test=[]
            min_test_loss = []
            
            archetypRange = np.arange(2,4+1,2)
            #for k in range(2,4+1,2):
            for k in archetypRange:
                # the loss tuple is (test_loss, test_loss_std ,min_loss based on seeds)
                loss_tuple = np.load(datapath + f"test_loss_{modality}_for_split-{split}_k-{k}.npy")
                means_test.append(loss_tuple[0])
                stds_test.append(loss_tuple[1])
                min_test_loss.append(loss_tuple[2])

            print(means_test)    
            
            plt.errorbar(archetypRange,means_test,yerr=stds_test,label=f"{modality}_test",linestyle="dashed",color=color_dict[modality])
            # plot the min test loss
            plt.plot(archetypRange,min_test_loss,label=f"{modality}_min_test",linestyle="dotted",color=color_dict[modality])
            # add to all train losses
            all_test_losses.append(means_test)

        #plot summarized test loss
        # sum over modalities
        all_test_losses = np.array(all_test_losses)
        all_test_losses = np.sum(all_test_losses,axis=0)

        plt.plot(archetypRange,all_test_losses,label="sum_test",color=color_dict["sum"],linestyle="dashed")         

        plt.legend()
        plt.title(f"Train and test loss for different number of archetypes split {split}")
        plt.xticks(archetypRange)
        plt.xlabel("Number of archetypes")
        plt.ylabel("Loss")
        plt.savefig(savepath + f"loss_split_{split}.png")        
        plt.close()

if __name__ == "__main__":    
    modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"], ["eeg", "fmri"], ["meg", "fmri"],["eeg"], ["meg"], ["fmri"]]
    for modalityComb in modalityCombs:        
            loss_pr_archetype_plot(path="data/MMAA_results/multiple_runs/",savepath="MMAA/plots/",modalityComb=modalityComb)
            
            




