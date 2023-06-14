import numpy as np
from pathlib import Path
from Multinomial_log_reg import train_LR
from pca import pca
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def confusion(y_true, y_predict):
    #make confusion matrix
    cm = confusion_matrix(y_true, y_predict)

    df_cm = pd.DataFrame(cm, index = ["true label: famous", "true label: scrambled", "true label: unfamiliar"],
                        columns = ["predicted: famous", "predicted: scrambled", "predicted: unfamiliar"])

    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

if __name__ == "__main__":
    trainPath = Path("data/trainingDataSubset")
    subjects = range(1,3)
    splits = range(0, 2)
    
    #one big list of true labels and predicted labels over all splits and seeds
    y_true = []
    y_predict = []
    
    #loop over all splits and seeds to get more data
    for split in splits:
        for seed in range(0, 100, 10):
            C = np.load(f"data/MMAA_results/multiple_runs/eeg-meg-fmri/split_{split}/C/C_split-{split}_k-20_seed-{seed}.npy")

            X_pca, y_train, i_var = pca(trainPath, subjects, C, plot = False, verbose=True)
            general_err_all, y_all_predicts, y_trues = train_LR(pca_data = True, multi = True, archetypes = 2, seed = 0)

            #append the true values and predicted values
            y_true.extend(np.asarray(y_trues).flatten())
            y_predict.extend(np.asarray(y_all_predicts).flatten())
    
    #plot it all
    confusion(y_true, y_predict)