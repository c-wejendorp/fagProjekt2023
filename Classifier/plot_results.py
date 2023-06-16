import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

from pathlib import Path
import numpy as np
from pca import pca
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression



def train_all(archetypes=2, seed=0,modalityComb=["eeg", "meg", "fmri"], reg_params=None, random_state = 0, datapath=None):
    
    trainPath = Path("data/trainingDataSubset")
    testPath = Path("data/testDataSubset")

    splits = range(2)

    subjects = range(1,17)
    subjects = ["sub-{:02d}".format(i) for i in subjects]

    conditions = ["famous", "scrambled", "unfamiliar"]
    
    # we'll just concatenate the results from split 0 and 1 inside this
    LR_general_err_all = defaultdict(lambda: [])
    # LR_y_all_predicts= []
    
    LR_pca_general_err_all = []
    # LR_pca_y_all_predicts= []
    baseline_general_err_all = []
    
    # y_trues = []
    for split in splits: 
        LR_pca_general_err_split = []
        baseline_general_err_split = []
        
        # Leave one out subject cross validation
        for test_subject_idx, test_subject in enumerate(subjects):
            all_subjects = range(1,17)
            train_subjects_idx = list(range(0,16))
            train_subjects_idx.remove(test_subject_idx)

            train_subjects = subjects[:]
            train_subjects.remove(test_subject)
            
            # C = np.load(f"data/MMAA_results/multiple_runs/eeg-meg-fmri/split_{split}/C/C_split-{split}_k-{archetypes}_seed-{seed}.npy")
            C = np.load(datapath + f"{'-'.join(modalityComb)}/split_{split}/C/C_split-{split}_k-{archetypes}_seed-{seed}.npy")
            
            # pca
            if split == 0:
                X, y, i_var = pca(trainPath, all_subjects, C, False, False, split)
            elif split == 1:
                X, y, i_var = pca(testPath, all_subjects, C, False, False, split)
            
            X = X[:,:(i_var+1)]
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
                    elif split == 1:
                        eeg_train_cond.append(np.load(testPath / f"{subject}/eeg/{condition}_test.npy"))
                        meg_train_cond.append(np.load(testPath / f"{subject}/meg/{condition}_test.npy"))
                
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
            
            ## Baseline
            #randomly choose labels as predictions
            baseline_pred = np.random.choice(np.unique(y_train), len(y_test))
            baseline_acc = np.sum(baseline_pred == y_test)/len(y_test)
            baseline_general_err_split.append(baseline_acc)
            
            
            ##  Train logistic regression
            # ___________no pca_____________
            
            for reg_param in reg_params:
                
                model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=1/reg_param, random_state=random_state)
                model_LR.fit(X_train, y_train)
                y_pred = model_LR.predict(X_test)
                
                acc = np.sum(y_pred == y_test)/len(y_test)
                
                # LR_y_all_predicts.append(y_pred)
                LR_general_err_all[reg_param].append(acc)
            
            # __________pca____________
            model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=random_state)
            model_LR.fit(X_pca_train, y_pca_train)
            y_pred = model_LR.predict(X_pca_test)
            
            acc = np.sum(y_pred == y_pca_test)/len(y_pca_test)
            
            # LR_pca_y_all_predicts.append(y_pred)
            # 
            # y_trues.append(y_test)
            # print("Accuracy:", acc)
            
            LR_pca_general_err_split.append(acc)
            
            
        # print(f"Generalization error split {split}: ", np.mean(LR_pca_general_err_split))
        LR_pca_general_err_all.append(np.mean(LR_pca_general_err_split))
        baseline_general_err_all.append(np.mean(baseline_general_err_split))
        
    reg_result_means = {reg_p: np.mean(accs) for reg_p, accs in LR_general_err_all.items()}
    # print("Done!")
    
    return reg_result_means, np.mean(LR_pca_general_err_all), np.mean(baseline_general_err_all)


def createLossPlot1(datapath = "data/MMAA_results/multiple_runs/", savepath = "Classifier/plots/",modalityComb=["eeg", "meg", "fmri"], reg_params=None, inp_archetype=2):
    
    # #datapath = Path(datapath) / Path(f"/{'-'.join(modalityComb)}/split_0/C/")   
    # C_datapath = datapath + f"{'-'.join(modalityComb)}/split_0/C/"   

    # # make save diractory
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
        
    # #open all files starting with eeg
    # # Look, this looks stupid, but bear with me: LR_loss[Archetype][Reg_param][seed_result]   :)))  (sorry)
    # LR_reg_ploss = defaultdict(lambda: defaultdict(lambda: []))  
    # LR_pca_loss = defaultdict(lambda: [])
    # baseline_loss = defaultdict(lambda: [])

    # for file in tqdm(os.listdir(C_datapath)): # I'm just going to assume that split_0 and split_1 has the same seeds and archetypes, if not, fight me >:(
    #     split, archetype, seed = re.findall(r'\d+', file)
    #     if not archetype == inp_archetype:
    #         continue
    #     reg_result_means, LR_pca_gen_acc, baseline_gen_acc = train_all(archetypes=archetype, seed=seed, reg_params=reg_params, modalityComb=modalityComb, datapath=datapath)
        
    #     LR_pca_loss[archetype].append(LR_pca_gen_acc)
    #     baseline_loss[archetype].append(baseline_gen_acc)
        
    #     for reg_p, mean_res in reg_result_means.items(): 
    #         LR_reg_ploss[archetype][reg_p].append(mean_res)
        
    #     f = open(f"Classifier/checkpoints_{'-'.join(modalityComb)}_k-{archetype}.txt", "a")
    #     print(f"____________Checkpoint archetype: {archetype}, seed: {seed}__________", file=f)
    #     print("baseline_loss = " + str(dict(baseline_loss)), file = f)
    #     print("LR_pca_loss = " + str(dict(LR_pca_loss)), file=f)
    #     print("LR_reg_ploss = " + str({k: dict(v) for k, v in dict(LR_reg_ploss).items()}), file=f)
    #     f.close()
    
    # # idk why, I just randomly call it loss instead of accuracy all the time
    # f = open(f"Classifier/results_{'-'.join(modalityComb)}_k-{inp_archetype}.txt", "a")
    # print("baseline_loss = " + str(dict(baseline_loss)), file = f)
    # print("LR_pca_loss = " + str(dict(LR_pca_loss)), file=f)
    # print("LR_reg_ploss = " + str({k: dict(v) for k, v in dict(LR_reg_ploss).items()}), file=f)
    # f.close()


    # # KNN_pca_loss = {'10': [0.38541666666666663, 0.3333333333333333, 0.32291666666666663, 0.40625, 0.3125, 0.3645833333333333, 0.4375, 0.35416666666666663, 0.3645833333333333, 0.32291666666666663], '12': [0.40625, 0.4375, 0.3645833333333333, 0.3333333333333333, 0.41666666666666663, 0.375, 0.38541666666666663, 0.34375, 0.3958333333333333, 0.34375], '14': [0.40625, 0.35416666666666663, 0.375, 0.3645833333333333, 0.40624999999999994, 0.39583333333333326]}
    # # KNN_loss = {'10': [0.38541666666666663, 0.41666666666666663, 0.3958333333333333, 0.46875, 0.44791666666666663, 0.38541666666666663, 0.44791666666666663, 0.375, 0.43749999999999994, 0.46875], '12': [0.3958333333333333, 0.3645833333333333, 0.40625, 0.41666666666666663, 0.40625, 0.3958333333333333, 0.38541666666666663, 0.4270833333333333, 0.38541666666666663, 0.44791666666666663], '14': [0.44791666666666663, 0.40625, 0.45833333333333326, 0.35416666666666663, 0.38541666666666663, 0.3958333333333333]}
    # # LR_pca_loss =  {'10': [0.43749999999999994, 0.38541666666666663, 0.4895833333333333, 0.40625, 0.38541666666666663, 0.41666666666666663, 0.40625, 0.44791666666666663, 0.44791666666666663, 0.375], '12': [0.3958333333333333, 0.34375, 0.375, 0.35416666666666663, 0.41666666666666663, 0.44791666666666663, 0.32291666666666663, 0.3958333333333333, 0.4583333333333333, 0.40625], '14': [0.41666666666666663, 0.40625, 0.4375, 0.375, 0.38541666666666663, 0.40625]}
    # # LR_loss =  {'10': [0.4583333333333333, 0.44791666666666663, 0.47916666666666663, 0.4583333333333333, 0.44791666666666663, 0.44791666666666663, 0.43749999999999994, 0.41666666666666663, 0.44791666666666663, 0.46875], '12': [0.40625, 0.4895833333333333, 0.44791666666666663, 0.44791666666666663, 0.48958333333333326, 0.41666666666666663, 0.4583333333333333, 0.4270833333333333, 0.45833333333333326, 0.44791666666666663], '14': [0.4895833333333333, 0.44791666666666663, 0.4375, 0.44791666666666663, 0.4375, 0.4583333333333333]}

    # LR_best = defaultdict(lambda: {})
    # LR_loss = defaultdict(lambda: [])
    # for archetype, reg_dict in LR_reg_ploss.items():
    #     LR_reg_results= {}
    #     for reg_p, seed_results in reg_dict.items():
    #         LR_reg_results[reg_p] = np.mean(seed_results)
        
    #     best_reg_param = max(LR_reg_results, key=LR_reg_results.get)
    #     best_reg_result = LR_reg_results[best_reg_param]
        
    #     LR_best[archetype][best_reg_param] = best_reg_result
    #     LR_loss[archetype].append(best_reg_result)
        
    # f = open(f"Classifier/results_{'-'.join(modalityComb)}_k-{inp_archetype}.txt", "a")
    # print("_________Best choice of regularization parameters and their results_________", file=f)
    # print("LR_best = " + str(dict(LR_best)), file=f)
    # print("LR_loss = "  + str(dict(LR_loss)), file=f)
    # f.close()
    
    baseline_loss = {'2': [0.3020833333333333, 0.40625, 0.375, 0.3020833333333333, 0.3020833333333333, 0.3958333333333333, 0.34375, 0.32291666666666663, 0.40625, 0.22916666666666663],
                     '4': [0.2708333333333333, 0.32291666666666663, 0.3333333333333333, 0.29166666666666663, 0.38541666666666663, 0.3125, 0.3125, 0.375, 0.3645833333333333, 0.40625],
                     '6': [0.2708333333333333, 0.32291666666666663, 0.375, 0.40625, 0.41666666666666663, 0.29166666666666663, 0.26041666666666663, 0.3645833333333333, 0.375, 0.3645833333333333],
                     '8': [0.38541666666666663, 0.35416666666666663, 0.3125, 0.3125, 0.3645833333333333, 0.375, 0.4270833333333333, 0.32291666666666663, 0.3020833333333333, 0.34375],
                     '10': [0.21875, 0.375, 0.3333333333333333, 0.34375, 0.44791666666666663, 0.23958333333333331, 0.2708333333333333, 0.29166666666666663, 0.4375, 0.24999999999999997],
                     '12': [0.3020833333333333, 0.42708333333333326, 0.32291666666666663, 0.35416666666666663, 0.32291666666666663, 0.3958333333333333, 0.40625, 0.3020833333333333, 0.3333333333333333, 0.3125],
                     '14': [0.3333333333333333, 0.26041666666666663, 0.32291666666666663, 0.3020833333333333, 0.32291666666666663, 0.3645833333333333, 0.375, 0.3333333333333333, 0.26041666666666663, 0.24999999999999997],
                     '16': [0.3020833333333333, 0.38541666666666663, 0.24999999999999997, 0.3333333333333333, 0.31249999999999994, 0.32291666666666663, 0.25, 0.26041666666666663, 0.47916666666666663, 0.32291666666666663],
                     '18': [0.3125, 0.26041666666666663, 0.40625, 0.29166666666666663, 0.375, 0.34375, 0.35416666666666663, 0.3645833333333333, 0.2708333333333333, 0.38541666666666663],
                     '20': [0.25, 0.34375, 0.375, 0.37499999999999994, 0.29166666666666663, 0.375, 0.29166666666666663, 0.3125, 0.35416666666666663, 0.37499999999999994],
                     '22': [0.3958333333333333, 0.19791666666666666, 0.38541666666666663, 0.4270833333333333, 0.29166666666666663, 0.3125, 0.32291666666666663, 0.3333333333333333, 0.3645833333333333, 0.3020833333333333],
                     '24': [0.3333333333333333, 0.3333333333333333, 0.36458333333333337, 0.41666666666666663, 0.3125, 0.32291666666666663, 0.29166666666666663, 0.41666666666666663, 0.34375, 0.3020833333333333],
                     '26': [0.34375, 0.3020833333333333, 0.3125, 0.3125, 0.32291666666666663, 0.41666666666666663, 0.2708333333333333, 0.36458333333333326, 0.28125, 0.3020833333333333],
                     '28': [0.22916666666666663, 0.3020833333333333, 0.40625, 0.35416666666666663, 0.34375, 0.34375, 0.3333333333333333, 0.40625, 0.3333333333333333, 0.28125],
                     '30': [0.3125, 0.28125, 0.3645833333333333, 0.35416666666666663, 0.28125, 0.37499999999999994, 0.34375, 0.28125, 0.375, 0.35416666666666663],
                     '32': [0.24999999999999997, 0.32291666666666663, 0.40625, 0.40625, 0.38541666666666663, 0.35416666666666663, 0.3645833333333333, 0.28125, 0.3333333333333333, 0.40624999999999994],
                     '34': [0.3020833333333333, 0.26041666666666663, 0.35416666666666663, 0.35416666666666663, 0.38541666666666663, 0.26041666666666663, 0.32291666666666663, 0.22916666666666663, 0.34375, 0.34375],
                     '36': [0.35416666666666663, 0.3020833333333333, 0.28125, 0.29166666666666663, 0.2708333333333333, 0.35416666666666663, 0.3020833333333333, 0.3333333333333333, 0.32291666666666663, 0.3125],
                     '38': [0.40625, 0.25, 0.3958333333333333, 0.375, 0.3125, 0.3020833333333333, 0.28125, 0.35416666666666663, 0.28125, 0.24999999999999997],
                     '40': [0.34375, 0.3125, 0.29166666666666663, 0.4270833333333333, 0.3645833333333333, 0.3333333333333333, 0.29166666666666663, 0.4270833333333333, 0.3020833333333333, 0.29166666666666663]
                     }
    LR_pca_loss = {'2': [0.41666666666666663, 0.4270833333333333, 0.41666666666666663, 0.41666666666666663, 0.41666666666666663, 0.41666666666666663, 0.41666666666666663, 0.41666666666666663, 0.4270833333333333, 0.41666666666666663],
                   '4': [0.40625, 0.43749999999999994, 0.4270833333333333, 0.4583333333333333, 0.44791666666666663, 0.4375, 0.46874999999999994, 0.4583333333333333, 0.40625, 0.44791666666666663],
                   '6': [0.45833333333333326, 0.5104166666666666, 0.41666666666666663, 0.5, 0.49999999999999994, 0.4375, 0.43749999999999994, 0.47916666666666663, 0.44791666666666663, 0.41666666666666663],
                   '8': [0.44791666666666663, 0.46875, 0.46875, 0.5104166666666666, 0.48958333333333326, 0.53125, 0.47916666666666663, 0.5, 0.44791666666666663, 0.48958333333333326],
                   '10': [0.5104166666666666, 0.46874999999999994, 0.5, 0.4583333333333333, 0.47916666666666663, 0.47916666666666663, 0.47916666666666663, 0.4583333333333333, 0.4895833333333333, 0.46874999999999994],
                   '12': [0.4270833333333333, 0.5104166666666666, 0.46874999999999994, 0.46874999999999994, 0.47916666666666663, 0.49999999999999994, 0.5, 0.47916666666666663, 0.5, 0.46874999999999994],
                   '14': [0.5, 0.47916666666666663, 0.5208333333333333, 0.46875, 0.5416666666666666, 0.48958333333333326, 0.48958333333333326, 0.47916666666666663, 0.44791666666666663, 0.47916666666666663],
                   '16': [0.48958333333333326, 0.46875, 0.49999999999999994, 0.4583333333333333, 0.44791666666666663, 0.5416666666666666, 0.49999999999999994, 0.5104166666666666, 0.5104166666666666, 0.53125],
                   '18': [0.5104166666666666, 0.49999999999999994, 0.5104166666666666, 0.5104166666666666, 0.4583333333333333, 0.47916666666666663, 0.5104166666666666, 0.5520833333333333, 0.5416666666666666, 0.47916666666666663],
                   '20': [0.5208333333333333, 0.5104166666666666, 0.53125, 0.46875, 0.49999999999999994, 0.4583333333333333, 0.42708333333333326, 0.53125, 0.48958333333333326, 0.47916666666666663],
                   '22': [0.53125, 0.53125, 0.5104166666666666, 0.44791666666666663, 0.53125, 0.5, 0.5104166666666666, 0.53125, 0.5208333333333333, 0.5208333333333333],
                   '24': [0.4895833333333333, 0.5104166666666666, 0.49999999999999994, 0.4895833333333333, 0.48958333333333326, 0.47916666666666663, 0.46875, 0.5416666666666666, 0.5104166666666666, 0.5416666666666666],
                   '26': [0.5208333333333333, 0.5, 0.5104166666666666, 0.5104166666666666, 0.5520833333333333, 0.53125, 0.53125, 0.53125, 0.5416666666666666, 0.5],
                   '28': [0.5520833333333333, 0.5625, 0.5208333333333333, 0.53125, 0.53125, 0.53125, 0.5104166666666666, 0.48958333333333326, 0.5416666666666666, 0.4895833333333333],
                   '30': [0.5208333333333333, 0.5104166666666666, 0.53125, 0.5104166666666666, 0.5520833333333333, 0.53125, 0.49999999999999994, 0.5208333333333333, 0.5, 0.5520833333333333],
                   '32': [0.5208333333333333, 0.5104166666666666, 0.5104166666666666, 0.53125, 0.5208333333333333, 0.5104166666666666, 0.4895833333333333, 0.5104166666666666, 0.49999999999999994, 0.5208333333333333],
                   '34': [0.5520833333333333, 0.49999999999999994, 0.5416666666666666, 0.5208333333333333, 0.5104166666666666, 0.53125, 0.5, 0.48958333333333326, 0.5416666666666666, 0.5208333333333333],
                   '36': [0.5104166666666666, 0.5104166666666666, 0.49999999999999994, 0.5104166666666666, 0.53125, 0.5104166666666666, 0.5104166666666666, 0.5104166666666666, 0.5104166666666666, 0.5],
                   '38': [0.48958333333333326, 0.5416666666666666, 0.49999999999999994, 0.5208333333333333, 0.5208333333333333, 0.5520833333333333, 0.5208333333333333, 0.49999999999999994, 0.53125, 0.5208333333333333],
                   '40': [0.53125, 0.5416666666666666, 0.48958333333333326, 0.5729166666666666, 0.5104166666666666, 0.53125, 0.4895833333333333, 0.5104166666666666, 0.5104166666666666, 0.47916666666666663]}
    LR_loss = {'2': [0.44791666666666663, 0.44791666666666663, 0.4375, 0.44791666666666663, 0.44791666666666663, 0.44791666666666663, 0.44791666666666663, 0.4375, 0.44791666666666663, 0.4375],
                    '4': [0.45833333333333326, 0.49999999999999994, 0.41666666666666663, 0.46875, 0.4895833333333333, 0.41666666666666663, 0.44791666666666663, 0.4270833333333333, 0.46874999999999994, 0.46875],
                    '6': [0.47916666666666663, 0.46875, 0.46875, 0.4270833333333333, 0.46874999999999994, 0.4583333333333333, 0.46874999999999994, 0.44791666666666663, 0.4479166666666667, 0.4166666666666667],
                    '8': [0.41666666666666663, 0.4375, 0.46875, 0.44791666666666663, 0.46875, 0.4583333333333333, 0.4895833333333333, 0.47916666666666663, 0.40625, 0.5104166666666666],
                    '10': [0.44791666666666663, 0.4375, 0.47916666666666663, 0.46875, 0.4375, 0.44791666666666663, 0.4583333333333333, 0.4375, 0.41666666666666663, 0.4583333333333333],
                    '12': [0.47916666666666663, 0.44791666666666663, 0.4895833333333333, 0.40625, 0.5104166666666666, 0.44791666666666663, 0.42708333333333326, 0.41666666666666663, 0.45833333333333337, 0.44791666666666663],
                    '14': [0.4583333333333333, 0.5, 0.46875, 0.46875, 0.4583333333333333, 0.44791666666666663, 0.41666666666666663, 0.4375, 0.43749999999999994, 0.44791666666666663],
                    '16': [0.45833333333333326, 0.44791666666666663, 0.42708333333333337, 0.4375, 0.45833333333333337, 0.4895833333333333, 0.4583333333333333, 0.44791666666666663, 0.41666666666666663, 0.46874999999999994],
                    '18': [0.4270833333333333, 0.4583333333333333, 0.4583333333333333, 0.4583333333333333, 0.41666666666666663, 0.44791666666666663, 0.44791666666666663, 0.4375, 0.43749999999999994, 0.4583333333333333],
                    '20': [0.44791666666666663, 0.4583333333333333, 0.4166666666666667, 0.46875, 0.4583333333333333, 0.4479166666666667, 0.47916666666666663, 0.44791666666666663, 0.47916666666666663, 0.45833333333333337],
                    '22': [0.4375, 0.4375, 0.45833333333333326, 0.4583333333333333, 0.4583333333333333, 0.44791666666666663, 0.4375, 0.46875, 0.45833333333333337, 0.4270833333333333],
                    '24': [0.4375, 0.46875, 0.44791666666666663, 0.47916666666666663, 0.46875, 0.4375, 0.44791666666666663, 0.4375, 0.4583333333333333, 0.42708333333333337],
                    '26': [0.4375, 0.44791666666666663, 0.44791666666666663, 0.44791666666666663, 0.46875, 0.44791666666666663, 0.4375, 0.46875, 0.46874999999999994, 0.42708333333333337],
                    '28': [0.4583333333333333, 0.47916666666666663, 0.44791666666666663, 0.4583333333333333, 0.4583333333333333, 0.47916666666666663, 0.42708333333333337, 0.45833333333333337, 0.4583333333333333, 0.4583333333333333],
                    '30': [0.42708333333333337, 0.44791666666666663, 0.44791666666666663, 0.44791666666666663, 0.4270833333333333, 0.44791666666666663, 0.45833333333333337, 0.44791666666666663, 0.44791666666666663, 0.44791666666666663],
                    '32': [0.44791666666666663, 0.46875, 0.42708333333333337, 0.46875, 0.46874999999999994, 0.46875, 0.45833333333333337, 0.4166666666666667, 0.44791666666666663, 0.44791666666666663],
                    '34': [0.45833333333333326, 0.4479166666666667, 0.4375, 0.42708333333333326, 0.41666666666666663, 0.4375, 0.46874999999999994, 0.4583333333333333, 0.44791666666666663, 0.45833333333333337],
                    '36': [0.4375, 0.42708333333333337, 0.4375, 0.4270833333333333, 0.44791666666666663, 0.44791666666666663, 0.44791666666666663, 0.4583333333333333, 0.42708333333333337, 0.44791666666666663],
                    '38': [0.46875, 0.4583333333333333, 0.42708333333333337, 0.44791666666666663, 0.44791666666666663, 0.47916666666666663, 0.42708333333333326, 0.4375, 0.4375, 0.4375],
                    '40': [0.47916666666666663, 0.4375, 0.4583333333333333, 0.46875, 0.4270833333333333, 0.44791666666666663, 0.4375, 0.4270833333333333, 0.4270833333333333, 0.4375]}

    best_params = [0.0001,0.001,0.01,0.1,0.1,0.1,0.1,0.01,0.1,0.001,0.1,0.1,0.1,0.1, 0.1,0.01,0.1,0.1,0.1,0.1]
    
    
    # calculate the mean and std for each number of archetypes 
    
    baseline_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in baseline_loss.items()], dtype="float64")
    baseline_std = np.array([np.std(loss) for archetype, loss in LR_pca_loss.items()])
    
    LR_pca_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_pca_loss.items()], dtype="float64")
    LR_pca_std = np.array([np.std(loss) for archetype, loss in LR_pca_loss.items()])
    
    LR_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_loss.items()], dtype="float64")
    LR_std = np.array([np.std(loss) for archetype, loss in LR_loss.items()])
    
    #plot the mean values with std
    plt.plot(LR_pca_mean[:,0], LR_pca_mean[:,1], '-', label = "LR_pca")
    plt.fill_between(LR_pca_mean[:,0], LR_pca_mean[:,1] - LR_pca_std, LR_pca_mean[:,1] + LR_pca_std, alpha=0.2)
    
    plt.plot(LR_mean[:,0], LR_mean[:,1], '-', label = "LR")
    plt.fill_between(LR_mean[:,0], LR_mean[:,1] - LR_std, LR_mean[:,1] + LR_std, alpha=0.2)
    
    plt.plot(baseline_mean[:,0], baseline_mean[:,1], '-', label = "Baseline")
    plt.fill_between(baseline_mean[:,0], baseline_mean[:,1] - baseline_std, baseline_mean[:,1] + baseline_std, alpha=0.2)
    
    
    # plt.errorbar(LR_pca_mean[:,0], LR_pca_mean[:,1], yerr = LR_pca_std, label = "LR_pca")
    # plt.errorbar(LR_mean[:,0], LR_mean[:,1], yerr = LR_std, label = "LR")
    # plt.errorbar(baseline_mean[:,0], baseline_mean[:,1], yerr = baseline_std, label = "Baseline")
    plt.legend()
    # make the x ticks integers 
    plt.xticks(LR_pca_mean[:,0])
    
    for a,b, param in zip(LR_mean[:,0], LR_mean[:,1], best_params): 
        plt.text(a, b, f"{param}")
        
    plt.title("Final classification loss for different number of archetypes training data")
    plt.xlabel("Number of archetypes")
    plt.ylabel("Generalization accuracy")
    plt.savefig(savepath + f"class_error__{'-'.join(modalityComb)}.txt.png")    
    plt.show()     

   
   
if __name__ == "__main__":
    reg_params = [1e-1, 1e-5]
    inp_archetype = "2"
    datapath = "data/MMAA_results/multiple_runs/"
    # data_path_HPC = "/work3/s204090/data/MMAA_results/multiple_runs/"
    createLossPlot1(datapath=datapath, modalityComb=["eeg", "meg", "fmri"], inp_archetype=inp_archetype, reg_params=reg_params)
    #close all plots
    plt.close("all")
