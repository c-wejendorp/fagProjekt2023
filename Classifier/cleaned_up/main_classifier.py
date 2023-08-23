import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
from collections import defaultdict
import yaml
import json
from pathlib import Path
import numpy as np
from pca import pca
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
from datetime import datetime

#NOTE: (multicond) should fmri be included in the training data for classification? It only really only adds noise no?



def train_all(archetypes=2, seed=0,modalityComb=["eeg", "meg", "fmri"], reg_params = [1e-3, 1e-1, 10], random_state = 0, datapath="data/MMAA_results/multiple_runs/", complexity_reducer='True', model_type = "multicond", subjects=range(1,17)):
    trainPath = Path("data/trainingDataSubset")
    testPath = Path("data/testDataSubset")
    
    splits = range(2)
    
    subjects_name = ["sub-{:02d}".format(i) for i in subjects]

    conditions = ["famous", "scrambled", "unfamiliar"]
    
    # we'll just concatenate the results from split 0 and 1 inside this
    LR_general_err_all = defaultdict(lambda: [])
    # LR_y_all_predicts= []
    LR_pca_general_err_all = []
    baseline_general_err_all = []
    
    # y_trues = []
    for split in splits: 
        LR_pca_general_err_split = []
        baseline_general_err_split = []
        
        # Leave one out subject cross validation
        for test_subject_idx, test_subject in enumerate(subjects_name):
            train_subjects_idx = list(subjects)
            train_subjects_idx.remove(test_subject_idx + 1) # +1 because subjects is 1 indiced

            train_subjects = subjects_name[:]
            train_subjects.remove(test_subject)
            
            
            X_train = []
            y_train = []

            X_test = []
            y_test = []
            
            # get S data
            if model_type == "spatconc":
                S = np.load(datapath + f"{'-'.join(modalityComb)}/split_{split}/Sms/Sms_split-{split}_k-{archetypes}_seed-{seed}.npy")

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
                    y_test.extend([condition]) 
                    
                    
                X_train = np.array(X_train)
                X_test = np.array(X_test)
                    
            elif model_type == "multicond":
                # get X and C to calculate the archetypes
                if split == 0:
                    train_inp_path = trainPath
                    train_inp_str = "train"
                    test_inp_path = testPath
                    test_inp = "test"
                elif split == 1:
                    train_inp_path = testPath
                    train_inp_str = "test"
                    test_inp_path = trainPath
                    test_inp = "train"
                    
                C = np.load(datapath + f"{'-'.join(modalityComb)}/split_{split}/C/C_split-{split}_k-{archetypes}_seed-{seed}.npy")
                
                for condition in conditions:
                    eeg_train_cond = []
                    meg_train_cond = []
                    
                    eeg_test_cond = []
                    meg_test_cond = []
                    
                    # Get train signal data
                    for subject in train_subjects:                             
                        eeg_train_cond.append(np.load(train_inp_path / f"{subject}/eeg/{condition}_{train_inp_str}.npy"))
                        meg_train_cond.append(np.load(train_inp_path / f"{subject}/meg/{condition}_{train_inp_str}.npy"))
                        
                            
                    # Get test signal data
                    eeg_test_cond.append(np.load(test_inp_path / f"{test_subject}/eeg/{condition}_{test_inp}.npy"))
                    meg_test_cond.append(np.load(test_inp_path / f"{test_subject}/meg/{condition}_{test_inp}.npy"))
                
                    # X_train.extend(np.concatenate([np.array(eeg_train_cond)@C, np.array(meg_train_cond)@C], axis=1)) 
                    # X_test.extend(np.concatenate([np.array(eeg_test_cond)@C, np.array(meg_test_cond)@C], axis=1))
                    
                    ## Debugging purposes
                    X_train.extend(np.concatenate([np.array(eeg_train_cond)@C[C.shape[0] // 3: 2 * C.shape[0] // 3, :], np.array(meg_train_cond)@C[C.shape[0] // 3: 2 * C.shape[0] // 3, :]], axis=1)) 
                    X_test.extend(np.concatenate([np.array(eeg_test_cond)@C[C.shape[0] // 3: 2 * C.shape[0] // 3, :], np.array(meg_test_cond)@C[C.shape[0] // 3: 2 * C.shape[0] // 3, :]], axis=1))
                    
                    y_test.extend([condition]) 
                    y_train.extend([condition] * len(train_subjects))
                
                X_train = np.array(X_train)
                X_test = np.array(X_test)
                
                X_train = X_train.reshape((X_train.shape[0],X_train.shape[1] * X_train.shape[2]))
                X_test = X_test.reshape((X_test.shape[0],X_test.shape[1] * X_test.shape[2]))


            
            # pca
            if complexity_reducer in ['pca', 'both']:
                X_pca_train, X_pca_test, y_pca_train, i_var = pca(nr_subjects=train_subjects_idx, plot=False, verbose=False, split=split, X_train=X_train, y_train=y_train, X_test=X_test)

                # Get enough components to explain 95% variance
                X_pca_train = X_pca_train[:,:(i_var+1)]
                X_pca_test = X_pca_test[:,:(i_var+1)]
                
                
            ## Baseline
            #randomly choose labels as predictions
            baseline_pred = np.random.choice(np.unique(y_train), len(y_test))
            baseline_acc = np.sum(baseline_pred == y_test)/len(y_test)
            baseline_general_err_split.append(baseline_acc)
            
            
            ##  Train logistic regression
            if complexity_reducer in ['pca', 'both']:
                # __________pca____________
                model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=random_state)
                model_LR.fit(X_pca_train, y_pca_train)
                y_pred = model_LR.predict(X_pca_test)
                
                acc = np.sum(y_pred == y_test)/len(y_test)
                
                LR_pca_general_err_split.append(acc)
            
            if complexity_reducer in ['regularization', 'both']:
                # ___________no pca_____________
                
                # NOTE: normal two layer cv could be used, but computing time would been horrible
                for reg_param in reg_params:
                    
                    model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=1/reg_param, random_state=random_state)
                    model_LR.fit(X_train, y_train)
                    y_pred = model_LR.predict(X_test)
                    
                    acc = np.sum(y_pred == y_test)/len(y_test)
                    
                    LR_general_err_all[reg_param].append(acc)
            
            
        # print(f"Generalization error split {split}: ", np.mean(LR_pca_general_err_split))
        if complexity_reducer in ['pca', 'both']:
            LR_pca_general_err_all.append(np.mean(LR_pca_general_err_split))
        baseline_general_err_all.append(np.mean(baseline_general_err_split))
        
    if complexity_reducer in ['regularization', 'both']:
        reg_result_means = {reg_p: np.mean(accs) for reg_p, accs in LR_general_err_all.items()}
    # print("Done!")
    elif complexity_reducer == 'pca': reg_result_means = None
    
    return reg_result_means, np.mean(LR_pca_general_err_all), np.mean(baseline_general_err_all)


def get_classifier_results(datapath = "data/MMAA_results/multiple_runs/", modalityComb=["eeg", "meg", "fmri"], reg_params=None, inp_archetype="2", output_path = "Classifier/results/test/", complexity_reducer='pca', random_state=0, model_type='multiconditional', subjects="all"):
    
    #datapath = Path(datapath) / Path(f"/{'-'.join(modalityComb)}/split_0/C/")
    if model_type == "spatial_concatenation":
        model_type = "spatconc"
        matrix_datapath = datapath + f"{'-'.join(modalityComb)}/split_0/Sms/"   # Doesn't matter whether we use split 1 or split 0 here, just need to extract archetypes and seeds
    elif model_type == "multiconditional":
        model_type = "multicond"
        matrix_datapath = datapath + f"{'-'.join(modalityComb)}/split_0/C/"
    else: 
        raise AssertionError(model_type + ": Not an available model_type. Try multiconditional/spatial_concatenation")
    
    
    if subjects == "all": subjects=range(1,17)
    
    
    # Look, this looks stupid, but bear with me: LR_loss[Archetype][Reg_param][seed](([classification_acc_list_indx])) :>
    LR_reg_ploss = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))  # .... sorry :^)
    LR_pca_loss = defaultdict(lambda: defaultdict(lambda: [])) # LR_loss[Archetype][seed](([classification_acc_list_indx]))
    baseline_loss = defaultdict(lambda: defaultdict(lambda: [])) 

    for file in tqdm(os.listdir(matrix_datapath)): # I'm just going to assume that split_0 and split_1 have the same seeds and archetypes, if not, fight me >:(
        split, archetype, seed = re.findall(r'\d+', file)
        if not archetype == inp_archetype:
            continue
        reg_result_means, LR_pca_gen_acc, baseline_gen_acc = train_all(archetypes=archetype, 
                                                                       seed=seed, 
                                                                       reg_params=reg_params, 
                                                                       modalityComb=modalityComb, 
                                                                       datapath=datapath, 
                                                                       complexity_reducer=complexity_reducer, 
                                                                       random_state=random_state, 
                                                                       model_type=model_type, 
                                                                       subjects=subjects)
        LR_pca_loss[archetype][seed].append(LR_pca_gen_acc)
        baseline_loss[archetype][seed].append(baseline_gen_acc)
        
        if complexity_reducer in ['regularization', 'both']:
            for reg_p, mean_res in reg_result_means.items(): 
                LR_reg_ploss[archetype][reg_p][seed].append(mean_res)
        
        
        # Ever wanted a list in a dictionairy in a dictionairy in a dictionairy in ANOTHER dictionairy? Too bad!
        json_output = {"LR_pca_loss": LR_pca_loss, "baseline_loss": baseline_loss, "LR_reg_ploss": LR_reg_ploss}
        
        # This is just a failsafe so everything can be extracted
        with open(output_path + f"checkpoints_{'-'.join(modalityComb)}_k-{archetype}.txt", "a") as f:
            json.dump(json_output, f, indent=4) 
    
        # Write final results to json
        with open(output_path + f"result{'-'.join(modalityComb)}_k-{archetype}.txt", "w") as f:
            json.dump(json_output, f, indent=4) 

    
    
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

# Multiprocess Pool class does not accept default arguments, so I'm making this dumb overlay function >:(
def parallization_overlay(CONFIG, output_path, inp_archetype):
    
    try:
        get_classifier_results(datapath=CONFIG['data_input_path'], 
                        modalityComb=CONFIG['modality_combination'], 
                        inp_archetype=inp_archetype, reg_params=CONFIG['regularization_parameters'], 
                        complexity_reducer = CONFIG['complexity_reducer'], 
                        random_state = CONFIG['random_state'], 
                        model_type = CONFIG['model_type'],
                        output_path=output_path,
                        subjects = CONFIG['subjects'])

        return (inp_archetype, 'Success')
    except Exception as e:
        return (inp_archetype, 'Failed', str(e))
    
    ## Debugging purposes
    # get_classifier_results(datapath=CONFIG['data_input_path'], 
    #                 modalityComb=CONFIG['modality_combination'], 
    #                 inp_archetype=inp_archetype, reg_params=CONFIG['regularization_parameters'], 
    #                 complexity_reducer = CONFIG['complexity_reducer'], 
    #                 random_state = CONFIG['random_state'], 
    #                 model_type = CONFIG['model_type'],
    #                 output_path=output_path,
    #                 subjects = CONFIG['subjects'])
    
    # return (inp_archetype, 'Success')











if __name__ == "__main__":
    #TODO: inserting system arguments changes the config file
    
    # Load classification configs
    BASE_DIR = os.path.dirname(__file__)
    with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    
    lowest_nr_archetypes = CONFIG['lowest_nr_archetypes']
    max_nr_archetypes = CONFIG['max_nr_archetypes']
    increment_step_size = CONFIG['increment_step_size']
         
    output_path = CONFIG['raw_output_path']
    model_type = CONFIG['model_type']
    # make output directory such that results_path/model_type/$run-number_$date/
    output_path += model_type + "/"
    date = datetime.now().strftime("%m-%d-%Y")
    if not os.path.exists(output_path):
        output_path += "1_" + date + "/"
    else:
        output_path += str(len(os.listdir(output_path)) + 1) + "_" + date + "/" # why am I like this...
    os.makedirs(output_path)


    # Prep list of input archetypes
    if max_nr_archetypes % increment_step_size == 0:
        list_of_input_archetypes = range(lowest_nr_archetypes, max_nr_archetypes + increment_step_size, increment_step_size)
    else: 
        list_of_input_archetypes = range(lowest_nr_archetypes, max_nr_archetypes, increment_step_size)

    list_of_input_archetypes = map(str, list_of_input_archetypes)
    
    
    # parallelization with a multiple processes over archetype inputs, using all the cpu cores available. Can be further parallelized during the CV subjects
    pool = mp.Pool(mp.cpu_count())
    success_log = [pool.apply(parallization_overlay, args=(CONFIG, output_path, inp_archetype)) for inp_archetype in list_of_input_archetypes]
    
    print("Classification results done!")
    
    # datapath = "data/MMAA_results/multiple_runs/"
    # data_path_HPC = "/work3/s204090/data/MMAA_results/multiple_runs/"

