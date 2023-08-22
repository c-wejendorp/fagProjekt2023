import numpy as np
import matplotlib.pyplot as plt
import os

# for split 0 we want to plot the average NMI with error bars for each modality for each number of archetypes.
# In the same plot we will also show the best average NMI across splits for EEG and MEG. (fmri is shared in both splits) 
def plotNMI(number_of_seeds = 10,mods = ["eeg", "meg", "fmri"],train=0,showPlot=False):
    
    # create folder if it does not exist    
    saveFolder = f"MMAA/plots/{'-'.join(mods)}/"
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)   

    colors = ["green", "red", "blue"]

    archetypRange = np.arange(2,20+1,2)

    #path = f"data/MMAA_results/multiple_runs/{'-'.join(mods)}/"
    if train == 1:
        test = 0
    else:
        test = 1   

    # lets start with split train

    datapath = f"data/MMAA_results/multiple_runs/{'-'.join(mods)}/split_{train}/NMI/"

    # loop over modalities
    offset = 0
    for idx, modality in enumerate(mods):
        NMI_tuples = []
        #now over archetypes
        
        for k in archetypRange:            
            NMI_tuples.append(np.load(datapath + f"NMI_{modality}_split-{train}_k-{k}.npy"))
            #NMI_tuples.append(np.load(datapath + f"NMI_split-{train}_k-{k}_type-{modality}.npy"))
        
        # now we have a list of tuples with (mean,std) for each k
        # we want to plot the mean with error bars
        NMI_mean = [t[0] for t in NMI_tuples]
        NMI_std = [t[1] for t in NMI_tuples]
        plt.errorbar(archetypRange+offset, NMI_mean, yerr=NMI_std, fmt='o', capsize=5,color=colors[idx], label=f"{modality}_split_{train}")
        offset += 0.3

    # now for the best NMI across splits for EEG and MEG
    # simple solution to not do this when mods is only fmri
    if mods != ["fmri"]:
    
        path = f"data/MMAA_results/multiple_runs/"

        offset = 0
        for idx, modality in enumerate(["eeg", "meg"]):
            NMI_best = []
            #now over archetypes
            for k in archetypRange:     
    
                NMI_best.append(max([np.load(f"data/MMAA_results/multiple_runs/{'-'.join(mods)}/split_{split}/NMI/NMI_{modality}_split-{split}_k-{k}.npy")[2] for split in [train, test]]))
                #NMI_best.append(max([np.load(path + f"split_{split}/NMI/NMI_split-{split}_k-{k}_type-{modality}.npy")[2] for split in [train, test]]))


            # plot it as a dotted line
            plt.plot(archetypRange+offset, NMI_best, '--', color=colors[idx], label=f"{modality}_best btw splits")
            
            offset += 0.3   

    #set the y axis ticks
    plt.yticks(np.arange(0.2,1.1,0.1))
    
    plt.xticks(archetypRange)   
    plt.legend(loc="lower right")
    #plt.legend()
    plt.xlabel("Number of archetypes")
    plt.ylabel("NMI")
    plt.title(f"Normalized Mutual Information, model: {'-'.join(modalityComb)}, split: {split}")
    plt.savefig(saveFolder + f"/NMI_split={train}.png", dpi=300)
    if showPlot:
        plt.show()
    plt.close()

if __name__ == "__main__":
    #modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"], ["eeg", "fmri"], ["meg", "fmri"],["eeg"], ["meg"], ["fmri"]]
    modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"], ["fmri"]]
    for modalityComb in modalityCombs:
        for split in [0,1]:
            plotNMI(mods=modalityComb,train=split,showPlot=False)
            
        
    


