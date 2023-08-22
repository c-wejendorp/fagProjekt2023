import numpy as np
import json
from nmi import nmi
import os

# this script and calc_test_loss_Sms could easily be fused
# as they have almost identical structure. i guess they've
# been kept separate for the sake of structure

if __name__ == "__main__":
    
    # we assume that all models are trained with the same seeds and splits and stepsize
    with open('MMAA/HPC/arguments/arguments0.json') as f:
            arguments = json.load(f)  

    seeds = arguments.get("seeds")    

    modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"], ["eeg", "fmri"], ["meg", "fmri"],["eeg"], ["meg"], ["fmri"]]

    # loop through all desired modality combinations
    for modalityComb in modalityCombs:
        
        #TODO: multicondition case + path
        
        # check the folder exists dir
        path = f'/work3/s204090/data/MMAA_results/multiple_runs/{"-".join(modalityComb)}/'
        if os.path.exists(path): 
            
            # loop over split 
            for split in [0,1]:
                 
                datapath = path + f'/split_{split}/'
                savepath = datapath + '/NMI/'
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                # loop over all archetypes in correct stepSize
                for numArcheTypes in range(2, 20+1, 2):
                    # load the average S matrix for each seed
                    S_matrices = []
                    for seed in seeds:
                        S_matrices.append(
                            np.load(datapath + f"S/S_split-{split}_k-{numArcheTypes}_seed-{seed}_sub-avg.npy"))         
                    
                    # calculate the NMI for S1 and S2, S2 and S3, etc and last S10 and S1 for each modality

                    # create dict based on modalityCom 
                    NMIS = {f"NMI_{modality}": [] for modality in modalityComb}

                    number_of_seeds = len(seeds)
                    for s in range(number_of_seeds):

                        # add to dict through list comprehension
                        for idx, modality in enumerate(modalityComb):
                            NMIS[f"NMI_{modality}"].append(nmi(S_matrices[s][idx], S_matrices[(s+1)%number_of_seeds][idx]))
                    
                    # calculate the average NMI for each modality
                    for NMI_name, nmis in NMIS.items():
                        # calculate the average, std and max NMI for each modality
                        mean = np.mean(nmis)
                        std = np.std(nmis)
                        max = np.max(nmis)
                        
                        #save the mean, std and max NMI for each modality'
                        np.save(savepath + f'{NMI_name}_split-{split}_k-{numArcheTypes}', np.array([mean, std, max]))