import json
import sys
import sys
from loadData import Real_Data
from MMAA_model_CUDA import MMAA, trainModel
import os
import numpy as np
import torch
from tqdm import tqdm

def train_archetypes(numArcheTypes):
    """trains the MM(M)AA model and saves the matrices and losses to files
    depending on the number of archetypes and seeds you provided

    Args:
        numArcheTypes (int): how many archetypes should be optimzed for in the MM(M)AA analysis
    """
    # TODO: I see some issues with global variables in this document.
    # perhaps fix and update function description
    
    # clear the cache
    torch.cuda.empty_cache()
    # print(f"Training model with {numArcheTypes} archetypes and seed {seed}")
    
    # run the MM(M)AA model 
    C, Sms, eeg_loss, meg_loss, fmri_loss, loss_Adam = trainModel(
            X,
            numArchetypes=numArcheTypes,
            seed=seed,
            plotDistributions=False,
            learningRate=1e-1,
            numIterations=arguments.get("iterations"), 
            loss_robust=arguments.get("lossRobust"),
            modalities=modalities)
    
    # save the C and S-matrix
    np.save(save_path_Cs + f'C_split-{split}_k-{numArcheTypes}_seed-{seed}', C)
    np.save(save_path_Sms + f'Sms_split-{split}_k-{numArcheTypes}_seed-{seed}', Sms)

    assert len(modalities) == Sms.shape[0], "The number of modalities does not match the number of modalities in the S matrix"            
    m, sub, k, _ = Sms.shape
    
    # # if one wishes to save all S matrices so subjects are split up
    # # this however, takes loads of memory
    # # filename for sub: S_split-x_k-x_seed-x_sub-x_mod-m
    # for i in range(m):
    #     for j in range(sub):
    #         np.save(save_path_SprSub + f'S_split-{split}_k-{k}_seed-{seed}_sub-{j}_mod-{modalities[i]}', Sms[i,j,:,:])

    # save an S-matrix where subjects have been averaged
    S_avg = np.mean(Sms, axis = 1)
    np.save(save_path_Ss + f'S_split-{split}_k-{k}_seed-{seed}_sub-avg', S_avg)

    # save all the losses
    # filename: loss_split-x_k-x_seed-x_type-m
    # m will be, eeg,meg,fmri and sum. 
    # sum is the sum of the three losses
    loss = [eeg_loss, meg_loss, fmri_loss, loss_Adam]
    loss_type = ['eeg', 'meg', 'fmri', 'sum']
    for i in range(len(loss)):
        if i == 3:
            # save the summed loss separately
            np.save(save_path_loss + f'loss_split-{split}_k-{k}_seed-{seed}_type-{loss_type[i]}', np.array(loss[i]))
        else:
            # save the eeg, meg and fmri loss in separate files  
            np.save(save_path_loss + f'loss_split-{split}_k-{k}_seed-{seed}_type-{loss_type[i]}', np.array([int(x.cpu().detach().numpy()) for x in loss[i]]))  

if __name__ == "__main__":  
    
    # get the split and argumentNum (which argument file to read) from from the command line
    # if these two are not given the program will stop
    assert len(sys.argv) == 3, f"Give split and argumentNum as command line arguments. Received: {sys.argv}"
    split = int(sys.argv[1])
    argumentsNum = int(sys.argv[2])

    # load arguments from json file
    with open(f'MMAA/HPC/arguments/arguments{argumentsNum}.json') as f:
        arguments = json.load(f)
    
    modalities = arguments.get("modalities")
        #arguments.get("subjects")
    
    # load data
    X = Real_Data(subjects=range(1,17),split=split)
    
    # define paths to everything worth saving
    work_path = "/work3/s204090/"
    
    save_path = work_path + f'data/MMAA_results/multiple_runs/{"-".join(modalities)}/split_{split}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path_Cs = save_path + 'C/'
    if not os.path.exists(save_path_Cs):
        os.makedirs(save_path_Cs)

    save_path_Ss = save_path + 'S/'
    if not os.path.exists(save_path_Ss):
        os.makedirs(save_path_Ss)

    save_path_SprSub = save_path + 'SprSub/'
    if not os.path.exists(save_path_SprSub):
        os.makedirs(save_path_SprSub)
    
    save_path_loss = save_path + 'loss/'
    if not os.path.exists(save_path_loss):
        os.makedirs(save_path_loss) 
    
    save_path_Sms = save_path + 'Sms/'
    if not os.path.exists(save_path_Sms):
        os.makedirs(save_path_Sms)

    # loop over seeds
    # print the torch default device
    print(torch.cuda.current_device())
    for seed in tqdm(arguments.get("seeds")):
        for numArcheTypes in range(arguments.get("archeTypeIntevalStart"),arguments.get("archeTypeIntevalStop")+1, arguments.get("archeTypeStepSizeStart")):
            train_archetypes(numArcheTypes=numArcheTypes)