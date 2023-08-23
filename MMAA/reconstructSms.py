import numpy as np
import os
import tqdm

def reconstructMatrix(path_to_multiple_runs="", modalityComb=[], k=0, seed=0):
    """saves the S-matrices from split 0 and 1 as an (m, s, k, v) matrix with a given
    seed value. 
    
    path_to_multiple_runs (str): parent path to all data
    modalityComb (list of str): which modalities were included in the analysis
    k (int): default=0. the number of archetypes in the analysis
    seed (int): default=0. the seed value used in the analysis"""
    
    
    # path to which modalities were included in the analysis
    modelDir = path_to_multiple_runs + f"{'-'.join(modalityComb)}/"
    
    # loop over splits
    for split in [0,1]:
        # define paths
        split_dir = modelDir + f"split_{split}/"
        S_dir = split_dir + "SprSub/"        

        Sms = []

        # loop over the modalities in the analysis
        for modality in modalityComb:             
            matrices_per_modalities = []
            
            #the matrices are saved as sub 0 to sub 15
            for j in range(16):
                # load each subject specific S-matrix
                S_subject = np.load(S_dir + f'S_split-{split}_k-{k}_seed-{seed}_sub-{j}_mod-{modality}.npy')
                matrices_per_modalities.append(S_subject)
            
            Sms.append(matrices_per_modalities)
        
        # save the Sms and create folder if not existing
        Sms = np.array(Sms)
        if not os.path.exists(split_dir + f'Sms/'):
            os.makedirs(split_dir + f'Sms/')
        
        # debug: check if shape is correct
        # print("This is the shape of Sms: ",Sms.shape,file=sys.stderr)
        # assert Sms.shape[:2] == [len(modalityComb),16, k]

        np.save(split_dir + f'Sms/Sms_split-{split}_k-{k}_seed-{seed}', Sms)

if __name__ == "__main__":
    
    path_to_multiple_runs = "data/MMAA_results/multiple_runs/"
    for modalityComb in [["eeg", "meg", "fmri"],["eeg", "meg"],]:
        archetypRange = np.concatenate((np.arange(2,16+1,2), np.arange(21, 76, 5)))
        
        for k in archetypRange:
            for s in tqdm.tqdm(range(0,10)):
                seed = s * 10
                reconstructMatrix(path_to_multiple_runs=path_to_multiple_runs, modalityComb=modalityComb, k=k, seed=seed)





    
