import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.linalg import svd
#import seaborn as sns
#from loadData import Real_Data

def pca(path, nr_subjects, C, plot = False, verbose = False, split=0):
    
    #load data
    trainPath = path

    subjects = nr_subjects
    subjects = ["sub-{:02d}".format(i) for i in subjects]
    conditions = ["famous", "scrambled", "unfamiliar"]

    #load c matrix for split 0 (training data)
    C = C

    X_train = np.array([])
    y_train = np.array([])

    #load in the ERP's for each condition an concatenate everything
    for subject in subjects: 
        for condition in conditions:
            eeg_train_cond = []
            meg_train_cond = []
            
            if split == 0:
                eeg_train_cond.append(np.load(trainPath / f"{subject}/eeg/{condition}_train.npy"))
                meg_train_cond.append(np.load(trainPath / f"{subject}/meg/{condition}_train.npy"))
            elif split == 1:
                eeg_train_cond.append(np.load(trainPath / f"{subject}/eeg/{condition}_test.npy"))
                meg_train_cond.append(np.load(trainPath / f"{subject}/meg/{condition}_test.npy"))
            #append archetypes and labels ERP's to training
            signal = np.concatenate((np.array(eeg_train_cond)@C, np.array(meg_train_cond)@C), axis=1)
            y_train = np.append(y_train, condition)
        
            #concatenate ERP's to one long feature vector
            X_train = np.append(X_train, np.concatenate(signal).reshape(-1, order = "F"))

    #reshape: [s*cond, t*k*2] matrix
    X_train = np.reshape(X_train, ((len(subjects) * len(conditions)), np.concatenate(signal).reshape(-1).shape[0]))

    # #equivalent way of loading the data. both are correct
    # X_train = np.array([])
    # y_train = np.array([])

    # #load in the ERP's for each condition an concatenate everything
    # X = Real_Data(range(1, 3))

    # A_eeg = X.EEG_data@C
    # A_meg = X.MEG_data@C
    
    # X_train_final = []
    # for subject in nr_subjects:
    #     for cond in range(3):
    #         erp_eeg = A_eeg[subject - 1][cond * 180:180 + cond * 180][:]
    #         erp_meg = A_meg[subject - 1][cond * 180:180 + cond * 180][:]
    #         erp = np.concatenate((erp_eeg, erp_meg), axis = 0)
    #         X_train_final.append(erp.reshape((-1), order = "F"))
    # X_train = np.asarray(X_train_final)
    
    #standardize
    mu = np.mean(X_train,axis=0,dtype=np.float64) 
    std = np.std(X_train,axis=0,dtype=np.float64)
    
    X_train_final = (X_train - mu)
    
    n = len(X_train_final)
    pca = PCA(n_components=n)

    #find the number of components need to exceed 95% variance for each subject
    X_pca = pca.fit_transform(X_train_final)
    pca.explained_variance_ratio_.cumsum()
    
    # #equivalent way of doing all this
    # #compute svd
    # U, S, Vh = svd(X_train_final, full_matrices = False)
    # V = Vh.T #[5400, 6]
    
    # #project observations onto eigenvector space
    # X_pca_svd = X_train_final @ V
    
    # #compute variance explained
    # rho = S * S / (S * S).sum()

    for i, var in enumerate(pca.explained_variance_ratio_.cumsum()):
        if var >= 0.95:
            i_var = i
            if verbose:
                print(f"explained variance exceeded 95% at {var} for {i + 1} components")
            break

    if plot:
        #plot ERP's as an average over the subjects
        fig, ax = plt.subplots(3, sharex = True)
        fig.suptitle("Centered ERP's for each condition")
        ax[0].plot(np.mean(X_train_final[np.arange(0, len(subjects)*len(conditions), 3),:], axis = 0), alpha = 0.3, color = "red", label = "famous")
        ax[1].plot(np.mean(X_train_final[np.arange(1, len(subjects)*len(conditions), 3),:], axis = 0), alpha = 0.3, color = "green", label = "scrambled")
        ax[2].plot(np.mean(X_train_final[np.arange(2, len(subjects)*len(conditions), 3),:], axis = 0), alpha = 0.3, color = "purple", label = "nonfamous")
        ax[0].set_ylim([-0.0006, 0.0006])
        ax[1].set_ylim([-0.0006, 0.0006])
        ax[2].set_ylim([-0.0006, 0.0006])
        ax[0].legend(loc = "upper right")
        ax[1].legend(loc = "upper right")
        ax[2].legend(loc = "upper right")
        ax[0].vlines(x = np.arange(0, X_train_final.shape[1], 180), ymin = -0.003, ymax = 0.003, linestyle = "dashed", color = "gainsboro")
        ax[1].vlines(x = np.arange(0, X_train_final.shape[1], 180), ymin = -0.003, ymax = 0.003, linestyle = "dashed", color = "gainsboro")
        ax[2].vlines(x = np.arange(0, X_train_final.shape[1], 180), ymin = -0.003, ymax = 0.003, linestyle = "dashed", color = "gainsboro")
        plt.show()
        
        #plot an how the first principal component looks
        fig = plt.figure()
        plt.plot(np.arange(X_train_final.shape[1]), pca.components_[0,:], alpha = 0.5, color = "lightblue", label = "pc1")
        plt.ylim([-0.1, 0.1])
        plt.legend(loc = "upper right")
        plt.vlines(x = np.arange(0, X_train_final.shape[1], 180), ymin = -0.06, ymax = 0.06, linestyle = "dashed", color = "gainsboro")
        plt.show()
        
        #plot how the observations are being projected
        _, ax = plt.subplots()
        for i, (color, condition) in enumerate(zip(['tab:blue', 'tab:orange', 'tab:green'], ["famous", "unfamous", "scrambled"])):
            x = [np.arange(X_pca.shape[1]) for _ in range(len(subjects))]
            y = X_pca[np.arange(i, len(subjects)*len(conditions), 3),:]

            plt.scatter(x, y, c=color, label=condition,
                    alpha=0.8, edgecolors='none')
            #plt.plot(x, y)

        ax.legend()
        plt.show()
        
        #plot explained variance:
        _ = plt.figure()
        ax = plt.axes()
        ax.set_title("Explained variance")
        
        plt.plot(np.arange(n), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='-', color='pink')
        plt.ylim(0.0,1.1)
        plt.xlabel('Number of Principal Components')
        plt.xticks(np.arange(n, step=1)) 
        plt.ylabel('Cumulative variance (%)')
        
        plt.axhline(y=0.95, color='grey', linestyle='--')
        plt.text(1.1, 1, '95% threshold', color = 'black', fontsize=16)

        ax.grid(axis='x')
        plt.tight_layout()
        
        plt.show()

        #plot datapoints on first three pc's
        _ = plt.figure()
        ax = plt.axes(projection = "3d")

        ax.set_title('Plotting on 3 PCs')

        # Set axes label
        ax.set_xlabel('pc1', labelpad=20)
        ax.set_ylabel('pc2', labelpad=20)
        ax.set_zlabel('pc3', labelpad=20)
        
        color = ["red", "blue", "green"]
        for i in range(len(conditions)):
            x = X_pca[np.arange(i, len(subjects)*len(conditions), 3), 0]
            y = X_pca[np.arange(i, len(subjects)*len(conditions), 3), 1]
            z = X_pca[np.arange(i, len(subjects)*len(conditions), 3), 2]

            ax.scatter(x, y, z, c = color[i])
        plt.show()

    return X_pca, y_train, i_var

if __name__ == "__main__":
    trainPath = Path("data/trainingDataSubset")
    subjects = range(1,3)
    C = np.load("data/MMAA_results/multiple_runs/eeg-meg-fmri/split_0/C/C_split-0_k-2_seed-0.npy")
    
    pca(trainPath, subjects, C, plot = True, verbose=True)