import numpy as np
import os
import tqdm
import sys

def reconstructMatrix(path_to_multiple_runs="", modalityComb=[], k=0, seed=0):

    modelDir = path_to_multiple_runs + f"{'-'.join(modalityComb)}/"
    # loop over splits
    for split in [0,1]:
        S_dir = modelDir + f"split_{split}/SprSub/"

        Sms = []

        for modality in modalityComb:             
            matrices_per_modalities = []
            #the matrices are saved as sub 0 to sub 15
            for j in range(16):
                S_subject = np.load(S_dir + f'S_split-{split}_k-{k}_seed-{seed}_sub-{j}_mod-{modality}.npy')
                matrices_per_modalities.append(S_subject)
            
            Sms.append(matrices_per_modalities)
        
        #save the Sms
        Sms = np.array(Sms)
        #crate folder called Sms
        if not os.path.exists(modelDir + f'Sms/'):
            os.makedirs(modelDir + f'Sms/')
        # check if shape is correct
        print("This is the shape of Sms: ",Sms.shape,file=sys.stderr)
        assert Sms.shape[:2] == [len(modalityComb),16, k]

        np.save(modelDir + f'Sms/Sms_split-{split}_k-{k}_seed-{seed}', Sms)

if __name__ == "__main__":
    path_to_multiple_runs = "/work3/s204090/data/MMAA_results/multiple_runs/"
    for modalityComb in [["eeg", "meg", "fmri"],["eeg", "meg"],]:
        for k in range(2,2+1,2):
        #for k in range(2,40+1,2):
            for s in tqdm.tqdm(range(0,10)):
                seed = s * 10
                reconstructMatrix(path_to_multiple_runs=path_to_multiple_runs, modalityComb=modalityComb, k=k, seed=seed)





    
