import numpy as np
import matplotlib.pyplot as plt
import os

def plotNMI(mods = ["eeg", "meg", "fmri"], train=0, showPlot=False):
    """plots the normalized mutual information (nmi) between the S-matrices
    for the three modalities and for each archetype. 
    
    mods (list of str): which modalities were included in the analysis
    train (int): which split was used as training data (0 or 1)
    showPlot (bool): default=False. if set to True, it will display the nmi plot
    """
    
    # create folder if it does not exist    
    saveFolder = f"MMAA/plots/{'-'.join(mods)}/"
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)   

    colors = ["green", "red", "blue"]

    archetypRange = np.arange(2, 20+1, 2)

    # test has opposite value of train
    if train == 1:
        test = 0
    else:
        test = 1   

    # define path to load nmi data
    datapath = f"data/MMAA_results/multiple_runs/{'-'.join(mods)}/split_{train}/NMI/"

    # loop over modalities
    offset = 0
    for idx, modality in enumerate(mods):
        NMI_tuples = []
        
        # loop over archetypes
        for k in archetypRange:            
            NMI_tuples.append(np.load(datapath + f"NMI_{modality}_split-{train}_k-{k}.npy"))
            
        # now we have a list of tuples with (mean, std) for each k
        # we want to plot the mean with error bars
        NMI_mean = [t[0] for t in NMI_tuples]
        NMI_std = [t[1] for t in NMI_tuples]
        
        # plot of the mean nmi for each modalitiy and archetype with error bars
        # horisontal offset to distinguish the colors (modalities)
        plt.errorbar(archetypRange + offset, NMI_mean, yerr=NMI_std, fmt='o', capsize=5, color=colors[idx], label=f"{modality}_split_{train}")
        offset += 0.3

    # TODO:
    # miscommunication was made here and simply plots the best nmi for each
    # archetype and split as opposed to plotting the nmi between the "best"
    # from split 0 and split 1 data (based perhaps on smallest loss). this
    # should of course be either changed or removed entirely
    
    # now for the best NMI across splits for EEG and MEG (fMRI was shared between splits)
    # simple solution to not do this when mods is only fmri
    if mods != ["fmri"]:
    
        path = f"data/MMAA_results/multiple_runs/"

        # only calculate nmi between eeg and meg
        offset = 0
        for idx, modality in enumerate(["eeg", "meg"]):
            NMI_best = []
            
            # find the greatest nmi across the splits for each archetypes
            for k in archetypRange: 
                NMI_best.append(max([np.load(path + f"{'-'.join(mods)}/split_{split}/NMI/NMI_{modality}_split-{split}_k-{k}.npy")[2] for split in [train, test]]))

            # plot the best nmi as a dotted line
            plt.plot(archetypRange+offset, NMI_best, '--', color=colors[idx], label=f"{modality}_best btw splits")
            
            # horizontal offset to distinguish the colors
            offset += 0.3   

    #set the y axis ticks
    plt.yticks(np.arange(0.2,1.1,0.1))
    
    plt.xticks(archetypRange)   
    plt.legend(loc="lower right")
    plt.xlabel("Number of archetypes")
    plt.ylabel("NMI")
    plt.title(f"Normalized Mutual Information, model: {'-'.join(modalityComb)}, split: {split}")
    plt.savefig(saveFolder + f"/NMI_split={train}.png", dpi=300)
    if showPlot:
        plt.show()
    plt.close()

if __name__ == "__main__":
    
    modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"], ["fmri"]]
    for modalityComb in modalityCombs:
        for split in [0,1]:
            plotNMI(mods=modalityComb,train=split,showPlot=False)
            
        
    


