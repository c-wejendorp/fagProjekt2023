import numpy as np
import os
import json
from loadData import Real_Data

if __name__ == "__main__":
    with open('MMAA/arguments.json') as f:
        arguments = json.load(f)    
    
    split = arguments.get("split")
    seeds = arguments.get("seeds")
    modalities = ['eeg', 'meg', 'fmri']

    datapath = f'/work3/s204090/data/MMAA_results/multiple_runs/split_{split}/'

    savepath = f'/work3/s204090/data/MMAA_results/multiple_runs/split_{split}/test_loss/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)       
    
    #S = np.load(f"data/MMAA_results/multiple_runs/split_{split}/S_split-{split}_k-2_seed-0_sub-avg.npy")

    
    #notice that x train can be either split 0 or 1
    if split == 0:
        X_train = Real_Data(subjects=arguments.get("subjects"),split=0)
        X_test = Real_Data(subjects=arguments.get("subjects"),split=1)
    else:
        X_train = Real_Data(subjects=arguments.get("subjects"),split=1)
        X_test = Real_Data(subjects=arguments.get("subjects"),split=0)
    
    #now loop over the different number of archetypes
    for numArcheTypes in range(arguments.get("archeTypeIntevalStart"),arguments.get("archeTypeIntevalStop")+1, arguments.get("archeTypeStepSize")):
        eeg_loss_testTrain = []
        meg_loss_testTrain = []
        fmri_loss_testTrain = []
        sum_loss_testTrain = []

        eeg_loss_testTest = []
        meg_loss_testTest = []
        fmri_loss_testTest = []
        sum_loss_testTest = []

        #calculate the loss for each seed        
        for seed in seeds:                
            S = np.load(datapath + f"S_split-{split}_k-{numArcheTypes}_seed-{seed}_sub-avg.npy")
            C = np.load(datapath + f"C_split-{split}_k-{numArcheTypes}_seed-{seed}_sub-avg.npy")

            #with testtrain
            eeg_loss_testTrain.append(np.linalg.norm(X_test[0] - np.linalg.multi_dot(X_train,C,S))**2)
            meg_loss_testTrain.append(np.linalg.norm(X_test[1] - np.linalg.multi_dot(X_train,C,S))**2)
            fmri_loss_testTrain.append(np.linalg.norm(X_test[2] - np.linalg.multi_dot(X_train,C,S))**2)
            sum_loss_testTrain.append(np.linalg.norm(X_test[0] - np.linalg.multi_dot(X_train,C,S))**2 + np.linalg.norm(X_test[1] - np.linalg.multi_dot(X_train,C,S))**2 + np.linalg.norm(X_test[2] - np.linalg.multi_dot(X_train,C,S))**2)
            #with testtest
            eeg_loss_testTest.append(np.linalg.norm(X_test[0] - np.linalg.multi_dot(X_test,C,S))**2)
            meg_loss_testTest.append(np.linalg.norm(X_test[1] - np.linalg.multi_dot(X_test,C,S))**2)
            fmri_loss_testTest.append(np.linalg.norm(X_test[2] - np.linalg.multi_dot(X_test,C,S))**2)
            sum_loss_testTest.append(np.linalg.norm(X_test[0] - np.linalg.multi_dot(X_test,C,S))**2 + np.linalg.norm(X_test[1] - np.linalg.multi_dot(X_test,C,S))**2 + np.linalg.norm(X_test[2] - np.linalg.multi_dot(X_test,C,S))**2)

            

          
            # calculate the average NMI for each modality
            eeg_loss_mean_testTrain = np.mean(eeg_loss_testTrain)
            meg_loss_mean_testTrain = np.mean(meg_loss_testTrain)
            fmri_loss_mean_testTrain = np.mean(fmri_loss_testTrain)
            sum_loss_mean_testTrain = np.mean(sum_loss_testTrain)

            eeg_loss_mean_testTest = np.mean(eeg_loss_testTest)
            meg_loss_mean_testTest = np.mean(meg_loss_testTest)
            fmri_loss_mean_testTest = np.mean(fmri_loss_testTest)
            sum_loss_mean_testTest = np.mean(sum_loss_testTest)

            # calculate the standard deviation for each modality
            eeg_loss_std_testTrain = np.std(eeg_loss_testTrain)
            meg_loss_std_testTrain = np.std(meg_loss_testTrain)
            fmri_loss_std_testTrain = np.std(fmri_loss_testTrain)
            sum_loss_std_testTrain = np.std(sum_loss_testTrain)

            eeg_loss_std_testTest = np.std(eeg_loss_testTest)
            meg_loss_std_testTest = np.std(meg_loss_testTest)
            fmri_loss_std_testTest = np.std(fmri_loss_testTest)
            sum_loss_std_testTest = np.std(sum_loss_testTest)

            
            

            #find the smallest loss for each modality
            eeg_loss_min = np.min(eeg_loss)
            meg_loss_min = np.min(meg_loss)
            fmri_loss_min = np.min(fmri_loss)

            #save the mean,std and max NMI for each modality'
            np.save(savepath + f'test_loss_for_split-{split}_k-{numArcheTypes}_type-eeg', np.array([eeg_loss_mean, eeg_loss_std, eeg_loss_min]))
            np.save(savepath + f'test_loss_for_split-{split}_k-{numArcheTypes}_type-meg', np.array([meg_loss_mean, meg_loss_std, meg_loss_min]))
            np.save(savepath + f'test_loss_for_split-{split}_k-{numArcheTypes}_type-fmri', np.array([fmri_loss_mean, fmri_loss_std, fmri_loss_min]))

            
           


           



            
        


        #load the C matrix



   
    
    

    

   





#datapath = "MMAA/data/MMAA_results/multiple_runs/split_0/"

def readLossFromFile(datapath,file,array = False):
    numArch = int(file.split("_")[1][4:])   
    if array:
        loss = np.load(datapath + file)   
    else :
        loss = np.load(datapath + file)[-1]

    modelSeed = int(file.split("_")[2][1:])

    return [numArch,loss,modelSeed]

def createLossPlot1(datapath = "data/MMAA_results/multiple_runs/split_0/",savepath = "MMAA/plots/"):
    #open all files starting with eeg
    eeg_loss = []
    meg_loss = []
    fmri_loss = []
    adam_loss = []
    for file in os.listdir(datapath):
        if file.startswith("eeg"):
            eeg_loss.append(readLossFromFile(datapath,file))            
        elif file.startswith("meg"):
            meg_loss.append(readLossFromFile(datapath,file))
        elif file.startswith("fmri"):
            fmri_loss.append(readLossFromFile(datapath,file))
        elif file.startswith("loss_adam"):
            adam_loss.append(readLossFromFile(datapath,file))

    #sort the lists
    eeg_loss.sort(key = lambda x: x[0])
    meg_loss.sort(key = lambda x: x[0])
    fmri_loss.sort(key = lambda x: x[0])
    adam_loss.sort(key = lambda x: x[0])

    # calculate the mean and std for each number of archetypes
    eeg_mean = np.mean([x[1] for x in eeg_loss])
    eeg_std = np.std([x[1] for x in eeg_loss])
    meg_mean = np.mean([x[1] for x in meg_loss])
    meg_std = np.std([x[1] for x in meg_loss])
    fmri_mean = np.mean([x[1] for x in fmri_loss])
    fmri_std = np.std([x[1] for x in fmri_loss])
    adam_mean = np.mean([x[1] for x in adam_loss])
    adam_std = np.std([x[1] for x in adam_loss])
    
    #plot the mean values with std
    plt.errorbar([x[0] for x in eeg_loss], [x[1] for x in eeg_loss], yerr = eeg_std, label = "eeg")
    plt.errorbar([x[0] for x in meg_loss], [x[1] for x in meg_loss], yerr = meg_std, label = "meg")
    plt.errorbar([x[0] for x in fmri_loss], [x[1] for x in fmri_loss], yerr = fmri_std, label = "fmri")
    plt.errorbar([x[0] for x in adam_loss], [x[1] for x in adam_loss], yerr = adam_std, label = "adam")
    plt.legend()
    # make the x ticks integers 
    plt.xticks([x[0] for x in eeg_loss])
    plt.title("Final loss for different number of archetypes training data")
    plt.xlabel("Number of archetypes")
    plt.ylabel("Final loss")
    plt.savefig(savepath + "finalLoss.png")    
    #plt.show()    

def createLossPlot2(datapath = "data/MMAA_results/multiple_runs/split_0/",savepath = "MMAA/plots/"):
    #open all files starting with eeg
    eeg_loss = []
    meg_loss = []
    fmri_loss = []
    adam_loss = []
    for file in os.listdir(datapath):
        if file.startswith("eeg"):
            eeg_loss.append(readLossFromFile(datapath,file,array = True))            
        elif file.startswith("meg"):
            meg_loss.append(readLossFromFile(datapath,file,array = True))
        elif file.startswith("fmri"):
            fmri_loss.append(readLossFromFile(datapath,file,array=True))
        elif file.startswith("loss_adam"):
            adam_loss.append(readLossFromFile(datapath,file,array=True))
    
    #sort the lists
    eeg_loss.sort(key = lambda x: x[0])
    meg_loss.sort(key = lambda x: x[0])
    fmri_loss.sort(key = lambda x: x[0])
    adam_loss.sort(key = lambda x: x[0])

    # for each number of archetypes, calculate the mean and std for each seed
    eeg_mean = []    
    meg_mean = []
    fmri_mean = []
    adam_mean = []
    
    for i in range(2,20+1,2):
        # the loss arrays are of different length
        # we we use the lenght of the shortest array
        # this is not the best solution, but it works
        #eeg
        eeg_losses=[x[1] for x in eeg_loss if x[0] == i]
        eeg_its=min([len(x) for x in eeg_losses])
        eeg_mean.append(np.mean([x[:eeg_its] for x in eeg_losses],axis = 0))

        #meg
        meg_losses=[x[1] for x in meg_loss if x[0] == i]
        meg_its=min([len(x) for x in meg_losses])
        meg_mean.append(np.mean([x[:meg_its] for x in meg_losses],axis = 0))

        #fmri
        fmri_losses=[x[1] for x in fmri_loss if x[0] == i]
        fmri_its=min([len(x) for x in fmri_losses])
        fmri_mean.append(np.mean([x[:fmri_its] for x in fmri_losses],axis = 0))

        #adam
        adam_losses=[x[1] for x in adam_loss if x[0] == i]
        adam_its=min([len(x) for x in adam_losses])
        adam_mean.append(np.mean([x[:adam_its] for x in adam_losses],axis = 0))
        
    
    #plot the loss curve for each number of archetypes
    for i in range(len(eeg_mean)):
    #for i in range(len(eeg_mean)):
        #if i > 8:
        plt.plot(range(len(eeg_mean[i])),eeg_mean[i],label = f"eeg_k={2*(i+1)}")
        plt.plot(range(len(meg_mean[i])),meg_mean[i],label = f"meg_k={2*(i+1)}")
        plt.plot(range(len(fmri_mean[i])),fmri_mean[i],label = f"fmri_k={2*(i+1)}")
        plt.plot(range(len(adam_mean[i])),adam_mean[i],label = f"adam_k={2*(i+1)}")
        

   
    plt.legend()
    # make the x ticks integers
    plt.title("Mean Loss curve for different number of archetypes")
    plt.xlabel("Iteration")
    plt.ylabel("Final loss")
    plt.savefig(savepath + "lossCurve.png")
    #plt.show()


    


   
   
if __name__ == "__main__":
    createLossPlot1()
    #close all plots
    plt.close("all")
    createLossPlot2()