import numpy as np
import os
import json
from loadData import Real_Data
import sys

if __name__ == "__main__":
    # we assume that all models are trained with the same seeds and splits and stepsize

    with open('MMAA/HPC/arguments/arguments0.json') as f:
            arguments = json.load(f)  

    seeds = arguments.get("seeds")    

    # here we do not include the one with only fmri 
    #modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"], ["eeg", "fmri"], ["meg", "fmri"],["eeg"], ["meg"]]    
    
    # for simple test of script. :
    modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"]]

    #loop through all modalities combinations
    for modalityComb in modalityCombs:
         #check the folder exists
        path = f'/work3/s204090/data/MMAA_results/multiple_runs/{"-".join(modalityComb)}/'
        if os.path.exists(path): 
            for split in [0,1]:
                # loop over split  
                datapath = path + f'/split_{split}/'

                datapath_C = datapath + 'C/'
                datapath_S = datapath + 'S/'
                
                savepath = datapath + 'test_loss/'
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                #notice that x train can be either split 0 or 1
                if split == 0:
                    X_train = Real_Data(subjects=arguments.get("subjects"),split=0)
                    X_test = Real_Data(subjects=arguments.get("subjects"),split=1)
                else:
                    X_train = Real_Data(subjects=arguments.get("subjects"),split=1)
                    X_test = Real_Data(subjects=arguments.get("subjects"),split=0)

                

                # loop over all archetypes in correct stepSize
                for numArcheTypes in range(arguments.get("archeTypeIntevalStart"),arguments.get("archeTypeIntevalStop")+1, arguments.get("archeTypeStepSize")):
                    # create dict based on modalityComb in compressed form except if modaility is fmri
                    modalities_loss = {f"test_loss_{modality}": [] for modality in modalityComb if modality != "fmri"}

                    #now find the loss for each seed 
                    for seed in seeds:
                        C = np.load(datapath_C + f"C_split-{split}_k-{numArcheTypes}_seed-{seed}.npy")
                        S = np.load(datapath_S + f"S_split-{split}_k-{numArcheTypes}_seed-{seed}_sub-avg.npy")

                        print("debugging",file=sys.stderr)
                        print(modalityComb,file=sys.stderr)
                        print(f"seed: {seed}, C: {C.shape}, S: {S.shape}",file=sys.stderr)
                        
                        
                        #calculate the loss for each modality
                        for idx,modality in enumerate(modalityComb): 
                                if modality != "fmri":                           
                                    org = getattr(X_test, f"{modality}_data")
                                    print(f"shape_org: {org}, C: {C.shape}",file=sys.stderr)
                                    XC = np.linalg.multi_dot([getattr(X_train, f"{modality}_data"),C])
                                    print(f"shape_XC: {XC}",file=sys.stderr)
                                    print(f"shape_S: {S[idx,:,:]}",file=sys.stderr)
                                    rec = np.linalg.multi_dot([XC,S[idx,:,:]])

                                    #rec = np.linalg.multi_dot([getattr(X_train, f"{modality}_data"),C,S[idx,:,:]])

                                    modalities_loss[f"test_loss_{modality}"].append(np.linalg.norm(org - rec)**2)

                    # calculate the mean and std over seeds for current num of archetypes
                    for key, loss_list in modalities_loss.items():
                        mean = np.mean(loss_list)
                        std = np.std(loss_list)
                        #find the smallest loss for each modality across seeds
                        min_loss = np.min(loss_list)
                        #save the mean,std and max NMI for each modality'
                        np.save(savepath + f'{key}_for_split-{split}_k-{numArcheTypes}', np.array([mean, std, min_loss]))
