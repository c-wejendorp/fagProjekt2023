import numpy as np
import matplotlib.pyplot as plt
import os
from KNN_classifier import train_KNN
from Multinomial_log_reg import train_LR
import re
from collections import defaultdict

"""
This can be optimized by SOOOOOO much by parallizing KNN and LR, but I need sleep,
so I'm too lazy. If you really need to run it, I would recommend to parallize before running
the code since it takes ages. Else I will parallize it when I get home from work - the train
functions are almost 1 to 1 the same.

"""



def createLossPlot1(datapath = "data/MMAA_results/multiple_runs/split_0/C/", savepath = "Classifier/plots/"):
    
    # make save diractory
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    #open all files starting with eeg
    KNN_loss = defaultdict(lambda: [])
    KNN_pca_loss = defaultdict(lambda: [])
    LR_loss = defaultdict(lambda: [])
    LR_pca_loss = defaultdict(lambda: [])

    for file in os.listdir(datapath): # I'm just going to assume that split_0 and split_1 has the same seeds and archetypes, if not, fight me >:(
        split, archetype, seed = re.findall(r'\d+', file)

        gen_err, _, _  = train_KNN(K_neighbors=10,distance_measure='Euclidean', pca_data=True, multi=True, archetypes=archetype, seed=seed)
        KNN_pca_loss[archetype].append(np.mean(gen_err))
        
        gen_err, _, _  = train_KNN(K_neighbors=10,distance_measure='Euclidean', pca_data=False, multi=True, archetypes=archetype, seed=seed)
        KNN_loss[archetype].append(np.mean(gen_err))
        
        gen_err, _, _  = train_LR(pca_data=True, multi=True, archetypes=archetype, seed=seed)
        LR_pca_loss[archetype].append(np.mean(gen_err))
        
        gen_err, _, _  = train_LR(pca_data=False, multi=True, archetypes=archetype, seed=seed)
        LR_loss[archetype].append(np.mean(gen_err))


    # calculate the mean and std for each number of archetypes 
    KNN_pca_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in KNN_pca_loss.items()])
    KNN_pca_std = np.array([np.std(loss) for archetype, loss in KNN_pca_loss.items()])
    
    KNN_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in KNN_loss.items()])
    KNN_std = np.array([np.std(loss) for archetype, loss in KNN_loss.items()])
    
    LR_pca_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_pca_loss.items()])
    LR_pca_std = np.array([np.std(loss) for archetype, loss in LR_pca_loss.items()])
    
    LR_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_loss.items()])
    LR_std = np.array([np.std(loss) for archetype, loss in LR_loss.items()])
    
    #plot the mean values with std
    plt.errorbar(KNN_pca_mean[:,0], KNN_pca_mean[:,1], yerr = KNN_pca_std, label = "KNN_pca")
    plt.errorbar(KNN_mean[:,0], KNN_mean[:,1], yerr = KNN_std, label = "KNN")
    plt.errorbar(LR_pca_mean[:,0], LR_pca_mean[:,1], yerr = LR_pca_std, label = "LR_pca")
    plt.errorbar(LR_mean[:,0], LR_mean[:,1], yerr = LR_std, label = "LR")
    plt.legend()
    # make the x ticks integers 
    plt.xticks(KNN_pca_mean[:,0])
    plt.title("Final classification loss for different number of archetypes training data")
    plt.xlabel("Number of archetypes")
    plt.ylabel("Final loss")
    plt.savefig(savepath + "class_error.png")    
    #plt.show()    

   
   
if __name__ == "__main__":
    createLossPlot1()
    #close all plots
    plt.close("all")
