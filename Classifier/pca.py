import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.linalg import svd

def pca(path, nr_subjects, C, plot = False, verbose = False, split=0):

    #load data
    trainPath = path

    subjects = nr_subjects
    subjects = ["sub-{:02d}".format(i) for i in subjects]
    conditions = ["famous", "scrambled", "unfamiliar"]

    #load c matrix for split 0 (training data)
    C = C

    X_train_final = np.array([])
    y_train_final = np.array([])

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
            y_train_final = np.append(y_train_final, condition)
        
            #concatenate ERP's to one long feature vector
            X_train_final = np.append(X_train_final, np.concatenate(signal).reshape(-1))

    #reshape: [s*cond, t*k*2] matrix
    X_train_final = np.reshape(X_train_final, ((len(subjects) * len(conditions)), np.concatenate(signal).reshape(-1).shape[0]))

    #standardize
    mu = np.mean(X_train_final,axis=0,dtype=np.float64) 
    std = np.std(X_train_final,axis=0,dtype=np.float64)

    X_train_final = (X_train_final - mu) / std
    
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
        #plot how the observations are being projected
        colors = ['r','k','b','lime','k','c','m','y','tab:purple','tab:pink','tab:gray','tab:orange','lime','tan','aquamarine','gold','lightgreen','tomato','papayawhip']
        fig = plt.figure()
        for i in range(len(conditions)):
            for j in range(len(subjects)):
                #plt.plot(range(V.shape[0]),V[:, 0], '-',color = colors[i], label = f"PC{i + 3 * j}") #plotting a principle component
                plt.plot(range(X_pca.shape[1]),X_pca[i + 3 * j,:], '-',color = colors[i], label = conditions[i])
        plt.legend()
        plt.xlabel('Principal component')
        plt.ylabel('Projected value')
        plt.show()

        #plot datapoints on first three pc's
        fig = plt.figure()
        ax = plt.axes(projection = "3d")

        ax.set_title('Plotting on 3 PCs')

        # Set axes label
        ax.set_xlabel('pc1', labelpad=20)
        ax.set_ylabel('pc2', labelpad=20)
        ax.set_zlabel('pc3', labelpad=20)

        for i in range(len(conditions)):
            x = X_pca[np.arange(i, len(subjects)*len(conditions), 3), 0]
            y = X_pca[np.arange(i, len(subjects)*len(conditions), 3), 1]
            z = X_pca[np.arange(i, len(subjects)*len(conditions), 3), 2]

            ax.scatter(x, y, z)
        plt.show()

    return X_pca, y_train_final, i_var

if __name__ == "__main__":
    trainPath = Path("data/trainingDataSubset")
    subjects = range(1,17)
    C = np.load(f"data/MMAA_results/split_0/C_matrix.npy")
    
    pca(trainPath, subjects, C, plot = True, verbose=True)