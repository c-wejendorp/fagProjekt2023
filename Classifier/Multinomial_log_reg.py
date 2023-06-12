
from pathlib import Path
import numpy as np
from pca import pca
from dtaidistance import dtw
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
# TODO: Make crossvalidation for subjects     

def train_LR(pca_data=True, multi=False, archetypes=None, seed=None):
    trainPath = Path("data/trainingDataSubset")
    testPath = Path("data/testDataSubset")

    model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    splits = range(2)

    subjects = range(1,17)
    subjects = ["sub-{:02d}".format(i) for i in subjects]

    conditions = ["famous", "scrambled", "unfamiliar"]
    general_err_all = []
    y_all_predicts= []
    y_trues = []
    for split in splits: 
        general_err_split = []
        
        # Leave one out subject cross validation
        for test_subject_idx, test_subject in tqdm(enumerate(subjects)):
            all_subjects = range(1,17)
            train_subjects_idx = list(range(0,16))
            train_subjects_idx.remove(test_subject_idx)

            train_subjects = subjects[:]
            train_subjects.remove(test_subject)
            
            if multi:
                if None in [archetypes, seed]:
                    print("please set nr of used archetypes and seed for running multi")
                    raise InterruptedError
                C = np.load(f"data/MMAA_results/multiple_runs/split_{split}/C/C_split-{split}_k-{archetypes}_seed-{seed}.npy")
            else:
                C = np.load(f"data/MMAA_results/split_{split}/C_matrix.npy")

            #S = np.load("data/MMAA_results/S_matrix.npy")
            
            ## if pca
            if pca_data:
                if split == 0:
                    X, y, i_var = pca(trainPath, all_subjects, C, False, False, split)
                elif split == 1:
                    X, y, i_var = pca(testPath, all_subjects, C, False, False, split)
                
                X = X[:,:i_var]
                X = X.reshape((len(all_subjects), 3, X.shape[1])) # 3 for nr number of conditions, let's hope this reshape is correct :))))))))))
                y = y.reshape((len(all_subjects), 3))
                # There is a better way to do this, but I'm tired, and I want sleep :)
                X_train = X[train_subjects_idx].reshape(((len(all_subjects)-1)*3, X.shape[2]))
                y_train = y[train_subjects_idx].reshape(((len(all_subjects)-1)*3,))
                
                X_test = X[test_subject_idx]
                y_test = y[test_subject_idx]
            
            ## if no pca:
            else:
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
        

                    
            if not pca_data:
                X_train = np.array(X_train)
                X_test = np.array(X_test)
                
                X_train = X_train.reshape((X_train.shape[0],X_train.shape[1] * X_train.shape[2]))
                X_test = X_test.reshape((X_test.shape[0],X_test.shape[1] * X_test.shape[2]))
                # Concatenate the subjects 
                # X_train = X_train.reshape((3*(len(subjects)-1),X_train.shape[1]*X_train.shape[2]))
                y_test = np.array(y_test)
                y_train = np.array(y_train)
            
            model_LR.fit(X_train, y_train)
            y_pred = model_LR.predict(X_test)
            
            acc = np.sum(y_pred == y_test)/len(y_test)
            
            y_all_predicts.append(y_pred)
            # print("Accuracy:", acc)
            
            general_err_split.append(acc)
        print(f"Generalization error split {split}: ", np.mean(general_err_split))
        general_err_all.append(np.mean(general_err_split))
        
    print("Overall generalization error:", np.mean(general_err_all))
    
    return general_err_all, y_all_predicts
    
if __name__ == '__main__':
    train_LR(pca_data = False, multi=True, archetypes=2, seed=0)
    train_LR(pca_data = False, multi=False)
    