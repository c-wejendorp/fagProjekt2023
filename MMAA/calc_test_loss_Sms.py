import numpy as np
import os
import json
from loadData import Real_Data
import sys
import torch

if __name__ == "__main__":

    #ensure that all tensors are on the GPU
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
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
        #path = f'data/MMAA_results/multiple_runs/{"-".join(modalityComb)}/'
        path = f'/work3/s204090/data/MMAA_results/multiple_runs/{"-".join(modalityComb)}/'
        if os.path.exists(path): 
            for split in [0,1]:
                # loop over split  
                datapath = path + f'/split_{split}/'

                datapath_C = datapath + 'C/'
                datapath_Sms = datapath + 'Sms/'
                
                savepath = datapath + 'test_loss_SMS/'
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                #notice that x train can be either split 0 or 1
                if split == 0:
                    X_train = Real_Data(subjects=arguments.get("subjects"),split=0)
                    X_test = Real_Data(subjects=arguments.get("subjects"),split=1)
                    T = np.array([getattr(X_train, f"{m}_data").shape[1] for m in modalityComb])
                else:
                    X_train = Real_Data(subjects=arguments.get("subjects"),split=1)
                    X_test = Real_Data(subjects=arguments.get("subjects"),split=0)
                    T = np.array([getattr(X_train, f"{m}_data").shape[1] for m in modalityComb])

                                

                # loop over all archetypes in correct stepSize
                for numArcheTypes in range(2,20+1,2):
                #for numArcheTypes in range(arguments.get("archeTypeIntevalStart"),arguments.get("archeTypeIntevalStop")+1, arguments.get("archeTypeStepSize")):
                    # create dict based on modalityComb in compressed form except if modaility is fmri
                    modalities_loss = {f"test_loss_{modality}": [] for modality in modalityComb if modality != "fmri"}

                    #now find the loss for each seed 
                    for seed in seeds:
                        C = np.load(datapath_C + f"C_split-{split}_k-{numArcheTypes}_seed-{seed}.npy")
                        Sms = np.load(datapath_Sms + f"Sms_split-{split}_k-{numArcheTypes}_seed-{seed}.npy")
                        #S = np.load(datapath_S + f"S_split-{split}_k-{numArcheTypes}_seed-{seed}_sub-avg.npy")

                        # C is the same for all subjects and modalities
                        C_tensor = torch.from_numpy(C)
                        C_tensor = C_tensor.double()                       

                        #print("debugging",file=sys.stderr)
                        #print(modalityComb,file=sys.stderr)
                        #print(f"seed: {seed}, C: {C.shape}, S: {S.shape}",file=sys.stderr)
                        
                        
                        #calculate the loss for each modality
                        for idx,modality in enumerate(modalityComb): 
                                if modality == "fmri":  
                                    continue                   
                             
                                # S is unique for each subject and modality
                                # remember S has the shape of (m, s, k, V)
                                S_tensor = torch.from_numpy(Sms[idx,:,:,:])
                                #S_tensor = torch.from_numpy(S[idx,:,:])                               
                                S_tensor = S_tensor.double()

                                X_test_tensor = torch.from_numpy(getattr(X_test, f"{modality}_data"))
                                X_train_tensor = torch.from_numpy(getattr(X_train, f"{modality}_data"))
                                #make all  tensors double
                                X_test_tensor = X_test_tensor.double()
                                X_train_tensor = X_train_tensor.double()
                                
                                #print("debugging",file=sys.stderr)
                                # print the current torch dtype
                                #print(f"X_test_tensor dtype: {X_test_tensor.dtype}",file=sys.stderr)
                                #print(f"X_train_tensor dtype: {X_train_tensor.dtype}",file=sys.stderr)
                                #print(f"C dtype: {C.dtype}",file=sys.stderr)
                                #print(f"S dtype: {S.dtype}",file=sys.stderr)                                                           


                                A = X_train_tensor@C_tensor
                                rec = A@S_tensor                             
                                loss_per_sub = torch.linalg.matrix_norm(X_test_tensor-rec)**2

                                # roboust loss
                                epsilon = 1e-3
                                beta  = 3/2 * epsilon
                                # find the highest number of time points across all modalities                                               
                                max_T = np.max(T)
                                alpha = 1 + max_T/2  - T[idx]/2
                                mle_loss_m = - (2 * (alpha + 1) + T[idx])/2 * torch.sum(torch.log(torch.add(loss_per_sub, 2 * beta)))
                                # cast tensor to cpu  and convert to numpy
                                mle_loss_m = mle_loss_m.cpu().detach().numpy()
                                
                                modalities_loss[f"test_loss_{modality}"].append(-mle_loss_m)                             
                
                    # calculate the mean and std over seeds for current num of archetypes
                    for key, loss_list in modalities_loss.items():
                        mean = np.mean(loss_list)
                        std = np.std(loss_list)
                        #find the smallest loss for each modality across seeds
                        min_loss = np.min(loss_list)
                        #save the mean,std and max NMI for each modality'
                        np.save(savepath + f'{key}_for_split-{split}_k-{numArcheTypes}', np.array([mean, std, min_loss]))
