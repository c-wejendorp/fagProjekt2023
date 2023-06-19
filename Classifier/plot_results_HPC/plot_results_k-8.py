import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

from pathlib import Path
import numpy as np
# from pca import pca
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression



def train_all(archetypes=2, seed=0,modalityComb=["eeg", "meg", "fmri"], reg_params=None, random_state = 0, datapath=None):
    splits = range(2)

    subjects = range(1,17)
    subjects = ["sub-{:02d}".format(i) for i in subjects]

    conditions = ["famous", "scrambled", "unfamiliar"]
    
    # we'll just concatenate the results from split 0 and 1 inside this
    LR_general_err_all = defaultdict(lambda: [])
    # LR_y_all_predicts= []

    baseline_general_err_all = []
    
    # y_trues = []
    for split in splits: 
        baseline_general_err_split = []
        
        # Leave one out subject cross validation
        for test_subject_idx, test_subject in enumerate(subjects):
            train_subjects_idx = list(range(0,16))
            train_subjects_idx.remove(test_subject_idx)

            train_subjects = subjects[:]
            train_subjects.remove(test_subject)
            
            S = np.load(datapath + f"{'-'.join(modalityComb)}/split_{split}/Sms/Sms_split-{split}_k-{archetypes}_seed-{seed}.npy")
            # No pca
            X_train = []
            y_train = []

            X_test = []
            y_test = []
            for condition in conditions:

                if condition == "famous":
                    S_cond = S[:,:,:,:18715]   # modality x subjects  x nr archetypes x nrconditions*nrsrouces
                elif condition == "scrambled":
                    S_cond = S[:,:,:,18715:2*18715]   
                elif condition == "unfamiliar":
                    S_cond = S[:,:,:,2*18715:]
                
                S_cond = np.reshape(S_cond, (S_cond.shape[0], S_cond.shape[1], S_cond.shape[2]*S_cond.shape[3]))
                for t_subject in train_subjects_idx:
                    X_train.append(np.concatenate([S_cond[0,t_subject,:], S_cond[1,t_subject,:]], axis=0)) #X_train.append(np.concatenate([np.mean(np.array(eeg_train_cond), axis=0)@C, np.mean(np.array(meg_train_cond), axis=0)@C])) # append the archetypes
                
                y_train.extend([condition] * len(train_subjects))
    
                X_test.append(np.concatenate([S_cond[0,test_subject_idx,:], S_cond[1,test_subject_idx,:]], axis=0))
                y_test.extend([condition]) # TODO make 360 general
                

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            
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
            
            
        # print(f"Generalization error split {split}: ", np.mean(LR_pca_general_err_split))
        baseline_general_err_all.append(np.mean(baseline_general_err_split))
        
    reg_result_means = {reg_p: np.mean(accs) for reg_p, accs in LR_general_err_all.items()}
    # print("Done!")
    
    return reg_result_means, np.mean(baseline_general_err_all)


def createLossPlot1(datapath = "data/MMAA_results/multiple_runs/", savepath = "Classifier/plots/",modalityComb=["eeg", "meg", "fmri"], reg_params=None, inp_archetype=2, savepath_res = "Classifier/plot_results_HPC/results_spatconc/"):
    
    #datapath = Path(datapath) / Path(f"/{'-'.join(modalityComb)}/split_0/C/")   
    S_datapath = datapath + f"{'-'.join(modalityComb)}/split_0/Sms/"   
    
    # make save diractory
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    if not os.path.exists(savepath_res):
        os.makedirs(savepath_res)
        
    #open all files starting with eeg
    # Look, this looks stupid, but bear with me: LR_loss[Archetype][Reg_param][seed_result]   :)))  (sorry)
    LR_reg_ploss = defaultdict(lambda: defaultdict(lambda: []))  
    baseline_loss = defaultdict(lambda: [])

    for file in tqdm(os.listdir(S_datapath)): # I'm just going to assume that split_0 and split_1 has the same seeds and archetypes, if not, fight me >:(
        split, archetype, seed = re.findall(r'\d+', file)
        if not archetype == inp_archetype:
            continue
        reg_result_means, baseline_gen_acc = train_all(archetypes=archetype, seed=seed, reg_params=reg_params, modalityComb=modalityComb, datapath=datapath)
        
        baseline_loss[archetype].append(baseline_gen_acc)
        
        for reg_p, mean_res in reg_result_means.items(): 
            LR_reg_ploss[archetype][reg_p].append(mean_res)
        
        f = open(savepath_res + f"checkpoints_{'-'.join(modalityComb)}_k-{archetype}.txt", "a")
        print(f"____________Checkpoint archetype: {archetype}, seed: {seed}__________", file=f)
        print("***baseline_loss***", file=f)
        print(str(dict(baseline_loss)), file = f)
        
        print("***LR_reg_ploss***", file=f)
        print(str({k: dict(v) for k, v in dict(LR_reg_ploss).items()}), file=f)
        f.close()
    
    # idk why, I just randomly call it loss instead of accuracy all the time
    f = open(savepath_res + f"results_{'-'.join(modalityComb)}_k-{inp_archetype}.txt", "a")
    print("****baseline_loss****", file=f)
    print(str(dict(baseline_loss)), file = f)
    print("*****LR_reg_ploss*****", file=f)
    print(str({k: dict(v) for k, v in dict(LR_reg_ploss).items()}), file=f)
    f.close()


    # KNN_pca_loss = {'10': [0.38541666666666663, 0.3333333333333333, 0.32291666666666663, 0.40625, 0.3125, 0.3645833333333333, 0.4375, 0.35416666666666663, 0.3645833333333333, 0.32291666666666663], '12': [0.40625, 0.4375, 0.3645833333333333, 0.3333333333333333, 0.41666666666666663, 0.375, 0.38541666666666663, 0.34375, 0.3958333333333333, 0.34375], '14': [0.40625, 0.35416666666666663, 0.375, 0.3645833333333333, 0.40624999999999994, 0.39583333333333326]}
    # KNN_loss = {'10': [0.38541666666666663, 0.41666666666666663, 0.3958333333333333, 0.46875, 0.44791666666666663, 0.38541666666666663, 0.44791666666666663, 0.375, 0.43749999999999994, 0.46875], '12': [0.3958333333333333, 0.3645833333333333, 0.40625, 0.41666666666666663, 0.40625, 0.3958333333333333, 0.38541666666666663, 0.4270833333333333, 0.38541666666666663, 0.44791666666666663], '14': [0.44791666666666663, 0.40625, 0.45833333333333326, 0.35416666666666663, 0.38541666666666663, 0.3958333333333333]}
    # LR_pca_loss =  {'10': [0.43749999999999994, 0.38541666666666663, 0.4895833333333333, 0.40625, 0.38541666666666663, 0.41666666666666663, 0.40625, 0.44791666666666663, 0.44791666666666663, 0.375], '12': [0.3958333333333333, 0.34375, 0.375, 0.35416666666666663, 0.41666666666666663, 0.44791666666666663, 0.32291666666666663, 0.3958333333333333, 0.4583333333333333, 0.40625], '14': [0.41666666666666663, 0.40625, 0.4375, 0.375, 0.38541666666666663, 0.40625]}
    # LR_loss =  {'10': [0.4583333333333333, 0.44791666666666663, 0.47916666666666663, 0.4583333333333333, 0.44791666666666663, 0.44791666666666663, 0.43749999999999994, 0.41666666666666663, 0.44791666666666663, 0.46875], '12': [0.40625, 0.4895833333333333, 0.44791666666666663, 0.44791666666666663, 0.48958333333333326, 0.41666666666666663, 0.4583333333333333, 0.4270833333333333, 0.45833333333333326, 0.44791666666666663], '14': [0.4895833333333333, 0.44791666666666663, 0.4375, 0.44791666666666663, 0.4375, 0.4583333333333333]}

    LR_best = defaultdict(lambda: {})
    LR_best_loss = defaultdict(lambda: [])
    LR_loss = defaultdict(lambda: [])
    for archetype, reg_dict in LR_reg_ploss.items():
        LR_reg_results= {}
        for reg_p, seed_results in reg_dict.items():
            LR_reg_results[reg_p] = np.mean(seed_results)
        
        best_reg_param = max(LR_reg_results, key=LR_reg_results.get)
        best_reg_result = LR_reg_results[best_reg_param]
        
        LR_best[archetype][best_reg_param] = best_reg_result
        LR_best_loss[archetype].append(best_reg_result)
        LR_loss[archetype] = LR_reg_ploss[archetype][best_reg_param]
    f = open(savepath_res + f"results_{'-'.join(modalityComb)}_k-{inp_archetype}.txt", "a")
    print("_________Best choice of regularization parameters and their results_________", file=f)
    print("LR_best = " + str(dict(LR_best)), file=f)
    print("LR_loss = "  + str(dict(LR_loss)), file=f)
    f.close()
    
    # # calculate the mean and std for each number of archetypes 
    # LR_pca_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_pca_loss.items()], dtype="float64")
    # LR_pca_std = np.array([np.std(loss) for archetype, loss in LR_pca_loss.items()])
    
    # LR_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_loss.items()], dtype="float64")
    # LR_std = np.array([np.std(loss) for archetype, loss in LR_loss.items()])
    
    # #plot the mean values with std
    # plt.errorbar(LR_pca_mean[:,0], LR_pca_mean[:,1], yerr = LR_pca_std, label = "LR_pca")
    # plt.errorbar(LR_mean[:,0], LR_mean[:,1], yerr = LR_std, label = "LR")
    # plt.legend()
    # # make the x ticks integers 
    # plt.xticks(LR_pca_mean[:,0])
    # plt.title("Final classification loss for different number of archetypes training data")
    # plt.xlabel("Number of archetypes")
    # plt.ylabel("Final loss")
    # plt.savefig(savepath + f"class_error__{'-'.join(modalityComb)}.txt.png")    
    # #plt.show()     

   
   
if __name__ == "__main__":
    reg_params = [10, 5, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    inp_archetype = "8"
    # datapath = "data/MMAA_results/multiple_runs/"
    data_path_HPC = "/work3/s204090/data/MMAA_results/multiple_runs/"
    createLossPlot1(datapath=data_path_HPC, modalityComb=["eeg", "meg", "fmri"], inp_archetype=inp_archetype, reg_params=reg_params)
    #close all plots
    plt.close("all")
