import json
import sys
import getopt
from loadData import Real_Data
from MMA_model_CUDA import MMAA, trainModel
import ast
import os
import numpy as np
import torch


if __name__ == "__main__":  
    #load arguments from json file
    with open('MMAA/arguments.json') as f:
        arguments = json.load(f)    

    split = arguments.get("split")
        
    X = Real_Data(subjects=arguments.get("subjects"),split=split)
    # loop over seeds
         
    save_path = f'/work3/s204090/data/MMAA_results/multiple_runs/split_{split}/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    save_path_SprSub = f'/work3/s204090/data/MMAA_results/multiple_runs/split_{split}/SprSub'
    if not os.path.exists(save_path_SprSub):
            os.makedirs(save_path_SprSub)
    
    save_path_loss = f'/work3/s204090/data/MMAA_results/multiple_runs/split_{split}/loss'
    if not os.path.exists(save_path_loss):
            os.makedirs(save_path_loss)

    

    for seed in arguments.get("seeds"):
        for numArcheTypes in range(arguments.get("archeTypeIntevalStart"),arguments.get("archeTypeIntevalStop")+1, arguments.get("archeTypeStepSize")):
            # lets clear the cache
            torch.cuda.empty_cache()
            #print(f"Training model with {numArcheTypes} archetypes and seed {seed}")
            C, Sms, eeg_loss, meg_loss, fmri_loss, loss_Adam = trainModel(
                    X,
                    numArchetypes=numArcheTypes,
                    seed=seed,
                    plotDistributions=False,
                    learningRate=1e-1,
                    numIterations=arguments.get("iterations"), 
                    loss_robust=arguments.get("lossRobust"))    
            
            # save the C matrix
            np.save(save_path + f'C_split-{split}_k-{numArcheTypes}_seed-{seed}', C)
    
            # save all the S matrices
            # filename for sub: S_split-x_k-x_seed-x_sub-x_mod-m
            # filename for average: S_split-x_k-x_seed-x_sub-avg
            modalities = ['eeg', 'meg', 'fmri']
            m,sub,k,_ = Sms.shape
            for i in range(m):
                for j in range(sub):
                    np.save(save_path_SprSub + f'S_split-{split}_k-{k}_seed-{seed}_sub-{j}_mod-{modalities[i]}', Sms[i,j,:,:])

            S_avg = np.mean(Sms, axis = 1)
            np.save(save_path + f'S_split-{split}_k-{k}_seed-{seed}_sub-avg', S_avg)

            # save all the losses
            # Save the different loss
            # filename: loss_split-x_k-x_seed-x_type-m
            # m will be, eeg,meg,fmri and sum. 
            # sum is the sum of the three losses
            loss = [eeg_loss, meg_loss, fmri_loss,loss_Adam]
            loss_type = ['eeg', 'meg', 'fmri', 'sum']
            for i in range(len(loss)):
                if i == 3:
                    np.save(save_path_loss + f'loss_split-{split}_k-{k}_seed-{seed}_type-{loss_type[i]}', np.array(loss[i]))
                else:    
                    np.save(save_path_loss + f'loss_split-{split}_k-{k}_seed-{seed}_type-{loss_type[i]}', np.array([int(x.cpu().detach().numpy())for x in loss[i]]))  


  
                  
            