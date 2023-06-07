
from pathlib import Path
import numpy as np

# TODO: Make crossvalidation for subjects 

class Nearest_Neighbor():
    def __init__(self, distance_measure = 'Euclidean', K_neighbors = 1):
        self.distance_measure = distance_measure
        self.K_neighbors = K_neighbors
        

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test_inp, y_test):
        distances = np.zeros(len(self.X_train))
        predicts = []
        for x_test in X_test_inp:
            for i, x in enumerate(self.X_train):
                if self.distance_measure == 'Euclidean':
                    distances[i] = np.sum(np.sqrt((x - x_test)**2))

            predicts.append(self.y_train[np.argmin(distances)])
            
        accuracy = np.sum(np.array(predicts) == np.array(y_test))/len(y_test)
        return predicts, accuracy
    

def main():
    trainPath = Path("data/trainingDataSubset")
    testPath = Path("data/testDataSubset")

    knn_clf=Nearest_Neighbor(distance_measure = 'Euclidean', K_neighbors = 1)

    splits = range(2)

    subjects = range(1,17)
    subjects = ["sub-{:02d}".format(i) for i in subjects]

    conditions = ["famous", "scrambled", "unfamiliar"]
    general_err = []
    y_all_predicts= []
    for split in splits: 
        C = np.load(f"data/MMAA_results/split_{split}/C_matrix.npy")
        #S = np.load("data/MMAA_results/S_matrix.npy")
        X_train = []
        y_train = []

        X_test = []
        y_test = []
        for condition in conditions:
            eeg_train_cond = []
            meg_train_cond = []
            
            eeg_test_cond = []
            meg_test_cond = []
            
            for subject in subjects: 
                if split == 0:
                    eeg_train_cond.append(np.load(trainPath / f"{subject}/eeg/{condition}_train.npy"))
                    meg_train_cond.append(np.load(trainPath / f"{subject}/meg/{condition}_train.npy"))

                    eeg_test_cond.append(np.load(testPath / f"{subject}/eeg/{condition}_test.npy"))
                    meg_test_cond.append(np.load(testPath / f"{subject}/meg/{condition}_test.npy"))
                
                elif split == 1:
                    eeg_test_cond.append(np.load(trainPath / f"{subject}/eeg/{condition}_train.npy"))
                    meg_test_cond.append(np.load(trainPath / f"{subject}/meg/{condition}_train.npy"))

                    eeg_train_cond.append(np.load(testPath / f"{subject}/eeg/{condition}_test.npy"))
                    meg_train_cond.append(np.load(testPath / f"{subject}/meg/{condition}_test.npy"))
                
            
            X_train.extend(np.concatenate([np.array(eeg_train_cond)@C, np.array(meg_train_cond)@C], axis=1)) #X_train.append(np.concatenate([np.mean(np.array(eeg_train_cond), axis=0)@C, np.mean(np.array(meg_train_cond), axis=0)@C])) # append the archetypes
            X_test.extend(np.concatenate([np.array(eeg_test_cond)@C, np.array(meg_test_cond)@C], axis=1))
            
            y_test.extend([condition] * len(subjects))
            y_train.extend([condition] * len(subjects))
            

        knn_clf.fit(X_train, y_train)
        y_pred, acc = knn_clf.predict(X_test, y_test)
        
        y_all_predicts.append(y_pred)
        print("Accuracy:", acc)
        
        general_err.append(acc)
        
    print("Generalization error:", np.mean(general_err))
    
    return general_err, y_all_predicts
    
if __name__ == '__main__':
    main()
    