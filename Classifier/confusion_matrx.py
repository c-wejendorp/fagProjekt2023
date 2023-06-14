import numpy as np
from Multinomial_log_reg import train_LR
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os

def confusion(y_true, y_predict, arch, pca = True, plot = False):
    #make confusion matrix
    cm = confusion_matrix(y_true, y_predict, labels = ["famous", "scrambled", "unfamiliar"])
    df_cm = pd.DataFrame(cm, index = ["true label: famous", "true label: scrambled", "true label: unfamiliar"],
                        columns = ["predicted: famous", "predicted: scrambled", "predicted: unfamiliar"])

    plt.figure(figsize = (10,7))
    plt.title(f"Confusion matrix for k = {arch}")
    sn.heatmap(df_cm, annot=True)
    if plot:
        plt.show()
    
    path = "Classifier/confusion_plots"
    # make save diractory
    if pca:
        path += "/pca"
    else:
        path += "/no_pca"
    if not os.path.exists(path):
        os.makedirs(path)
        
    plt.savefig(path + f"/k_{arch}")
    

if __name__ == "__main__":
    splits = range(0, 2)
    plot_pca = False
    
    for arch in range(2, 22, 2):
        #one big list of true labels and predicted labels over all splits and seeds
        y_true = []
        y_predict = []
        #loop over all splits and seeds to get more data
        for seed in range(0, 100, 10):
            general_err_all, y_all_predicts, y_trues = train_LR(pca_data = plot_pca, multi = True, archetypes = arch, seed = seed)

            #append the true values and predicted values
            y_true.extend(np.asarray(y_trues).flatten())
            y_predict.extend(np.asarray(y_all_predicts).flatten())

        #plot it all
        confusion(y_true, y_predict, arch, pca = plot_pca)