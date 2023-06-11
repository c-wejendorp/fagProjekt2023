import numpy as np
import json
from nmi import nmi
import os

if __name__ == "__main__":

    with open('MMAA/arguments.json') as f:
            arguments = json.load(f)  

    seeds = arguments.get("seeds")

    # loop over split
    for split in [0,1]:

        datapath = f'/work3/s204090/data/MMAA_results/multiple_runs/split_{split}/NMI/'
        if not os.path.exists(datapath):
            os.makedirs(datapath)       

        # loop over all archetypes in correct stepSize
        for numArcheTypes in range(arguments.get("archeTypeIntevalStart"),arguments.get("archeTypeIntevalStop")+1, arguments.get("archeTypeStepSize")):
            # load the average S matrix for each seed
            S_matrices=[]
            for seed in seeds:
                
                S_matrices.append(np.load(datapath + f"S_split-{split}_k-{numArcheTypes}_seed-{seed}_sub-avgnpy"))         
            # calculate the NMI for S1 and S2, S2 and S3, etc and last S10 and S1
            # this needs to be done for each modality
            eeg_NMIs = []
            meg_NMIs = []
            fmri_NMIs = []
            number_of_seeds = len(seeds)
            for s in range(number_of_seeds):
                eeg_NMIs.append(nmi(S_matrices[s][0], S_matrices[(s+1)%number_of_seeds][0]))
                meg_NMIs.append(nmi(S_matrices[s][1], S_matrices[(s+1)%number_of_seeds][1]))
                fmri_NMIs.append(nmi(S_matrices[s][2], S_matrices[(s+1)%number_of_seeds][2]))

            # calculate the average NMI for each modality
            eeg_NMI_mean = np.mean(eeg_NMIs)
            meg_NMI_mean = np.mean(meg_NMIs)
            fmri_NMI_mean = np.mean(fmri_NMIs)
            # calculate the standard deviation for each modality
            eeg_NMI_std = np.std(eeg_NMIs)
            meg_NMI_std = np.std(meg_NMIs)
            fmri_NMI_std = np.std(fmri_NMIs)

            #save the mean and std
            np.save(datapath + f'NMI_split-{split}_k-{numArcheTypes}_type-eeg', np.array([eeg_NMI_mean, eeg_NMI_std]))
            np.save(datapath + f'NMI_split-{split}_k-{numArcheTypes}_type-meg', np.array([meg_NMI_mean, meg_NMI_std]))
            np.save(datapath + f'NMI_split-{split}_k-{numArcheTypes}_type-fmri', np.array([fmri_NMI_mean, fmri_NMI_std]))






    

