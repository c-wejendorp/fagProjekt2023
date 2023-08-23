import numpy as np
import matplotlib.pyplot as plt

def loss_pr_archetype_plot(path = "data/MMAA_results/multiple_runs/", savepath="MMAA/plots/", modalityComb=["eeg","meg","fmri"]):
    """plots the training and test loss per modality as well as the sum of the three.
    
    path (str): path to retrieve loss data
    savepath (str): path to save the plots to
    modalityComb (list of str): which modalities were included in the analysis
    """
    
    # define paths
    folder = path + f'{"-".join(modalityComb)}/'
    savepath = savepath + f'{"-".join(modalityComb)}/'

    color_dict = {"eeg":"blue","meg":"red","fmri":"green","sum":"black"}
    archetypRange = np.arange(2,20+1,2)
    
    # loop over splits
    for split in [0,1]:
        # plot of training loss
        # note the training loss value has not been averaged over seeds
        datapath = folder + f"split_{split}/loss/"

        all_train_losses = []

        # loop over all modalities, archetypes and seeds
        for modality in modalityComb:
            means_train=[]
            stds_train=[]

            for k in archetypRange:
                loss_list = []
                for seed in range(0,91,10):
                    loss_pr_iterations=np.load(datapath + f"loss_split-{split}_k-{k}_seed-{seed}_type-{modality}.npy")
                    
                    # only append final training loss
                    loss_list.append(loss_pr_iterations[-1])
                
                # now find the mean and std
                means_train.append(np.mean(loss_list))
                stds_train.append(np.std(loss_list))
                
            # plot averaged training loss for all archetypes with err. bars
            plt.errorbar(archetypRange, means_train, yerr=stds_train, label=f"{modality} train", color=color_dict[modality], linestyle="solid")
            
            # add to all train losses
            if modality != "fmri":
                all_train_losses.append(means_train)       

        # plot summarized train loss 
        # check we do not only have fmri
        if modalityComb != ["fmri"]:
            # sum over modalities
            all_train_losses = np.array(all_train_losses)
            all_train_losses = np.sum(all_train_losses, axis=0)

            # plot the training loss summed over modalities
            plt.plot(archetypRange, all_train_losses, label="eeg+meg train", color=color_dict["sum"], linestyle="solid")
        
        # plot of test loss
        # does not make sense if we only have fmri
        if modalityComb != ["fmri"]:
            # update datapath
            # datapath = folder + f"split_{split}/test_loss/"
            datapath = folder + f"split_{split}/test_loss_SMS/" #TODO: where does the SMS come from here?

            all_test_losses = []
            
            # loop over modalities except fmri
            for modality in modalityComb:
                if modality == "fmri":
                    continue
                means_test=[]
                stds_test=[]
                min_test_loss = []
                
                # loop over archetypes
                for k in archetypRange:
                    # the loss tuple is (test_loss, test_loss_std ,min_loss based on seeds)
                    loss_tuple = np.load(datapath + f"test_loss_{modality}_for_split-{split}_k-{k}.npy")
                    
                    means_test.append(loss_tuple[0])
                    stds_test.append(loss_tuple[1])
                    min_test_loss.append(loss_tuple[2])

                # print(means_test)    
                
                # plot the mean test loss for the modalities with error bars
                plt.errorbar(archetypRange, means_test, yerr=stds_test, label=f"{modality} test", linestyle="dashed", color=color_dict[modality])
                
                # plot the min test loss
                #plt.plot(archetypRange,min_test_loss,label=f"{modality}_min_test",linestyle="dotted",color=color_dict[modality])
                
                # add to all train losses
                all_test_losses.append(means_test)

            # plot summed test loss
            # sum over modalities
            all_test_losses = np.array(all_test_losses)
            all_test_losses = np.sum(all_test_losses,axis=0)

            # plot test loss summed over modalities
            plt.plot(archetypRange, all_test_losses, label="eeg+meg test", color=color_dict["sum"], linestyle="dashed")         

        plt.legend()
        plt.title(f"Train and test loss pr number of archetypes, model: {'-'.join(modalityComb)}, split: {split}")
        plt.xticks(archetypRange)
        plt.xlabel("Number of archetypes")
        plt.ylabel("Loss")
        # increase the dpi for better quality
        plt.savefig(savepath + f"loss_split_{split}.png",dpi=300)        
        plt.close()

if __name__ == "__main__":    
    modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"], ["fmri"]]   
     
    for modalityComb in modalityCombs:        
        loss_pr_archetype_plot(path="data/MMAA_results/multiple_runs/",savepath="MMAA/plots/",modalityComb=modalityComb)
            
            




