import numpy as np
import matplotlib.pyplot as plt
import os
from KNN_classifier import train_KNN, Nearest_Neighbor
from Multinomial_log_reg import train_LR
import re
from collections import defaultdict

from pathlib import Path
import numpy as np
from pca import pca
from dtaidistance import dtw
#from tqdm import tqdm
from sklearn.linear_model import LogisticRegression



def train_all(K_neighbors=10,distance_measure='Euclidean', archetypes=2, seed=0,modalityComb=["eeg", "meg", "fmri"],pathToC="NoPathSet"):
    
    trainPath = Path("data/trainingDataSubset")
    testPath = Path("data/testDataSubset")

    splits = range(2)

    subjects = range(1,17)
    subjects = ["sub-{:02d}".format(i) for i in subjects]

    conditions = ["famous", "scrambled", "unfamiliar"]
    
    KNN_general_err_all = []
    # KNN_y_all_predicts= []
    
    KNN_pca_general_err_all = []
    # KNN_pca_y_all_predicts= []
    
    LR_general_err_all = []
    # LR_y_all_predicts= []
    
    LR_pca_general_err_all = []
    # LR_pca_y_all_predicts= []
    
    # y_trues = []
    for split in splits: 
        KNN_general_err_split = []
        KNN_pca_general_err_split = []
        LR_general_err_split = []
        LR_pca_general_err_split = []
        # Leave one out subject cross validation
        for test_subject_idx, test_subject in enumerate(subjects):
        #for test_subject_idx, test_subject in tqdm(enumerate(subjects)):
            
            all_subjects = range(1,17)
            train_subjects_idx = list(range(0,16))
            train_subjects_idx.remove(test_subject_idx)

            train_subjects = subjects[:]
            train_subjects.remove(test_subject)
            
            C = np.load(pathToC + f"/C_split-{split}_k-{archetypes}_seed-{seed}.npy")
            
            # pca
            if split == 0:
                X, y, i_var = pca(trainPath, all_subjects, C, False, False, split)
            elif split == 1:
                X, y, i_var = pca(testPath, all_subjects, C, False, False, split)
            
            X = X[:,:i_var]
            X = X.reshape((len(all_subjects), 3, X.shape[1])) # 3 for nr number of conditions, let's hope this reshape is correct :))))))))))
            y = y.reshape((len(all_subjects), 3))
            # There is a better way to do this, but I'm tired, and I want sleep :)
            X_pca_train = X[train_subjects_idx].reshape(((len(all_subjects)-1)*3, X.shape[2]))
            y_pca_train = y[train_subjects_idx].reshape(((len(all_subjects)-1)*3,))
            
            X_pca_test = X[test_subject_idx]
            y_pca_test = y[test_subject_idx]
            
            # No pca
            X_train = []
            y_train = []

            X_test = []
            y_test = []
            for condition in conditions:
                eeg_train_cond = []
                meg_train_cond = []
                
                eeg_test_cond = []
                meg_test_cond = []
                
                for subject in train_subjects: 
                    if split == 0:
                        eeg_train_cond.append(np.load(trainPath / f"{subject}/eeg/{condition}_train.npy"))
                        meg_train_cond.append(np.load(trainPath / f"{subject}/meg/{condition}_train.npy"))
                        
                        ## it is perhaps a bit of waste only to evaluate on a single subject when we have data for all of them. Uncomment if wanted to evaluate on all.
                        # eeg_test_cond.append(np.load(testPath / f"{subject}/eeg/{condition}_test.npy"))  
                        # meg_test_cond.append(np.load(testPath / f"{subject}/meg/{condition}_test.npy"))
                    
                    elif split == 1:
                        # eeg_test_cond.append(np.load(trainPath / f"{subject}/eeg/{condition}_train.npy"))
                        # meg_test_cond.append(np.load(trainPath / f"{subject}/meg/{condition}_train.npy"))

                        eeg_train_cond.append(np.load(testPath / f"{subject}/eeg/{condition}_test.npy"))
                        meg_train_cond.append(np.load(testPath / f"{subject}/meg/{condition}_test.npy"))
                
                ## Comment this if we want to evaluate on all test subjects
                if split == 0: 
                        eeg_test_cond.append(np.load(testPath / f"{test_subject}/eeg/{condition}_test.npy"))
                        meg_test_cond.append(np.load(testPath / f"{test_subject}/meg/{condition}_test.npy"))
                elif split == 1:
                        eeg_test_cond.append(np.load(trainPath / f"{test_subject}/eeg/{condition}_train.npy"))
                        meg_test_cond.append(np.load(trainPath / f"{test_subject}/meg/{condition}_train.npy"))
                
                
                
                X_train.extend(np.concatenate([np.array(eeg_train_cond)@C, np.array(meg_train_cond)@C], axis=1)) #X_train.append(np.concatenate([np.mean(np.array(eeg_train_cond), axis=0)@C, np.mean(np.array(meg_train_cond), axis=0)@C])) # append the archetypes
                X_test.extend(np.concatenate([np.array(eeg_test_cond)@C, np.array(meg_test_cond)@C], axis=1))
                
                y_test.extend([condition]) # TODO make 360 general
                y_train.extend([condition] * len(train_subjects))
    

                    

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            
            X_train = X_train.reshape((X_train.shape[0],X_train.shape[1] * X_train.shape[2]))
            X_test = X_test.reshape((X_test.shape[0],X_test.shape[1] * X_test.shape[2]))

            y_test = np.array(y_test)
            y_train = np.array(y_train)
            
            ##  Train logistic regression
            # no pca
            model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            model_LR.fit(X_train, y_train)
            y_pred = model_LR.predict(X_test)
            
            acc = np.sum(y_pred == y_test)/len(y_test)
            
            # LR_y_all_predicts.append(y_pred)
            LR_general_err_split.append(acc)
            
            # pca
            model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            model_LR.fit(X_pca_train, y_pca_train)
            y_pred = model_LR.predict(X_pca_test)
            
            acc = np.sum(y_pred == y_pca_test)/len(y_pca_test)
            
            # LR_pca_y_all_predicts.append(y_pred)
            # 
            # y_trues.append(y_test)
            # print("Accuracy:", acc)
            
            LR_pca_general_err_split.append(acc)
            
            ## KNN
            # pca
            model = Nearest_Neighbor(distance_measure=distance_measure, K_neighbors=K_neighbors)
            model.fit(X_train, y_train)
            y_pred, acc = model.predict(X_test, y_test)
            KNN_general_err_split.append(acc)
            # KNN_y_all_predicts.append(y_pred)
            # no pca
            model = Nearest_Neighbor(distance_measure=distance_measure, K_neighbors=K_neighbors)
            model.fit(X_pca_train, y_pca_train)
            y_pred, acc = model.predict(X_pca_test, y_pca_test)
            
            KNN_pca_general_err_split.append(acc)
            # KNN_pca_y_all_predicts.append(y_pred)
            
        # print(f"Generalization error split {split}: ", np.mean(LR_general_err_split))
        LR_general_err_all.append(np.mean(LR_general_err_split))
        
        # print(f"Generalization error split {split}: ", np.mean(LR_pca_general_err_split))
        LR_pca_general_err_all.append(np.mean(LR_pca_general_err_split))
        
        # print(f"Generalization error split {split}: ", np.mean(KNN_general_err_split))
        KNN_general_err_all.append(np.mean(KNN_general_err_split))
        
        # print(f"Generalization error split {split}: ", np.mean(KNN_pca_general_err_split))
        KNN_pca_general_err_all.append(np.mean(KNN_pca_general_err_split))
        
    print("Done!")
    
    return np.mean(LR_general_err_all), np.mean(LR_pca_general_err_all), np.mean(KNN_general_err_all), np.mean(KNN_pca_general_err_all)


def createLossPlot1(datapath = "data/MMAA_results/multiple_runs/", savepath = "Classifier/plots/",modalityComb=["eeg", "meg", "fmri"]):
    
    #datapath = Path(datapath) / Path(f"/{'-'.join(modalityComb)}/split_0/C/")   
    datapath = datapath + f"{'-'.join(modalityComb)}/split_0/C/"   

    # make save diractory
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    #open all files starting with eeg
    KNN_loss = defaultdict(lambda: [])
    KNN_pca_loss = defaultdict(lambda: [])
    LR_loss = defaultdict(lambda: [])
    LR_pca_loss = defaultdict(lambda: [])

    seeds = ["0","10"]
    for file in os.listdir(datapath):
        #for file in tqdm(os.listdir(datapath)): # I'm just going to assume that split_0 and split_1 has the same seeds and archetypes, if not, fight me >:(
        split, archetype, seed = re.findall(r'\d+', file)
        if seed not in seeds:
            continue
             
        LR_gen_acc, LR_pca_gen_acc, KNN_gen_acc, KNN_pca_gen_acc = train_all(K_neighbors=10, distance_measure='Euclidean', archetypes=archetype, seed=seed,modalityComb=modalityComb,pathToC=datapath)
        
        KNN_pca_loss[archetype].append(KNN_pca_gen_acc)
        KNN_loss[archetype].append(KNN_gen_acc)
        LR_pca_loss[archetype].append(LR_pca_gen_acc)
        LR_loss[archetype].append(LR_gen_acc)
        
        f = open(f"Classifier/checkpoints_{'-'.join(modalityComb)}_k-{archetype}.txt", "a")
        print(f"____________Checkpoint archetype: {archetype}, seed: {seed}__________", file=f)
        print("KNN_pca_loss = " + str(KNN_pca_loss), file=f)
        print("KNN_loss = " + str(KNN_loss), file=f)
        print("LR_pca_loss = " + str(LR_pca_loss), file=f)
        print("LR_loss = " + str(LR_loss), file=f)
        f.close()
    
    # idk why, I just randomly call it loss instead of accuracy all the time
    f = open(f"Classifier/results_{'-'.join(modalityComb)}_k-{archetype}.txt", "a")
    print("KNN_pca_loss = " + str(KNN_pca_loss), file=f)
    print("KNN_loss = " + str(KNN_loss), file=f)
    print("LR_pca_loss = " + str(LR_pca_loss), file=f)
    print("LR_loss = " + str(LR_loss), file=f)
    f.close()


    # KNN_pca_loss = {'10': [0.38541666666666663, 0.3333333333333333, 0.32291666666666663, 0.40625, 0.3125, 0.3645833333333333, 0.4375, 0.35416666666666663, 0.3645833333333333, 0.32291666666666663], '12': [0.40625, 0.4375, 0.3645833333333333, 0.3333333333333333, 0.41666666666666663, 0.375, 0.38541666666666663, 0.34375, 0.3958333333333333, 0.34375], '14': [0.40625, 0.35416666666666663, 0.375, 0.3645833333333333, 0.40624999999999994, 0.39583333333333326]}
    # KNN_loss = {'10': [0.38541666666666663, 0.41666666666666663, 0.3958333333333333, 0.46875, 0.44791666666666663, 0.38541666666666663, 0.44791666666666663, 0.375, 0.43749999999999994, 0.46875], '12': [0.3958333333333333, 0.3645833333333333, 0.40625, 0.41666666666666663, 0.40625, 0.3958333333333333, 0.38541666666666663, 0.4270833333333333, 0.38541666666666663, 0.44791666666666663], '14': [0.44791666666666663, 0.40625, 0.45833333333333326, 0.35416666666666663, 0.38541666666666663, 0.3958333333333333]}
    # LR_pca_loss =  {'10': [0.43749999999999994, 0.38541666666666663, 0.4895833333333333, 0.40625, 0.38541666666666663, 0.41666666666666663, 0.40625, 0.44791666666666663, 0.44791666666666663, 0.375], '12': [0.3958333333333333, 0.34375, 0.375, 0.35416666666666663, 0.41666666666666663, 0.44791666666666663, 0.32291666666666663, 0.3958333333333333, 0.4583333333333333, 0.40625], '14': [0.41666666666666663, 0.40625, 0.4375, 0.375, 0.38541666666666663, 0.40625]}
    # LR_loss =  {'10': [0.4583333333333333, 0.44791666666666663, 0.47916666666666663, 0.4583333333333333, 0.44791666666666663, 0.44791666666666663, 0.43749999999999994, 0.41666666666666663, 0.44791666666666663, 0.46875], '12': [0.40625, 0.4895833333333333, 0.44791666666666663, 0.44791666666666663, 0.48958333333333326, 0.41666666666666663, 0.4583333333333333, 0.4270833333333333, 0.45833333333333326, 0.44791666666666663], '14': [0.4895833333333333, 0.44791666666666663, 0.4375, 0.44791666666666663, 0.4375, 0.4583333333333333]}




    # calculate the mean and std for each number of archetypes 
    KNN_pca_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in KNN_pca_loss.items()], dtype="float64")
    KNN_pca_std = np.array([np.std(loss) for archetype, loss in KNN_pca_loss.items()])
    
    KNN_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in KNN_loss.items()], dtype="float64")
    KNN_std = np.array([np.std(loss) for archetype, loss in KNN_loss.items()])
    
    LR_pca_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_pca_loss.items()], dtype="float64")
    LR_pca_std = np.array([np.std(loss) for archetype, loss in LR_pca_loss.items()])
    
    LR_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_loss.items()], dtype="float64")
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
    plt.savefig(savepath + f"class_error__{'-'.join(modalityComb)}.txt.png")    
    #plt.show()    

   
   
if __name__ == "__main__":
    createLossPlot1(datapath="/work3/s204090/data/MMAA_results/multiple_runs/", modalityComb=["eeg", "meg", "fmri"])
    #close all plots
    plt.close("all")
