import torch 
import numpy as np
#import matplotlib.pyplot as plt
from loadData import Real_Data
#from tqdm import tqdm
import os

#ensure that all tensors are on the GPU
if torch.cuda.is_available():
    torch.set_default_device("cuda:0")
    
class MMAA(torch.nn.Module):
    def __init__(self, X: Real_Data, k : int, loss_robust: bool, modalities=["eeg", "meg", "fmri"]):
    #def __init__(self, V, T, k, X: Real_Data, numSubjects = 1, numModalities = 1): #k is number of archetypes
        super(MMAA, self).__init__()
        self.modalities = modalities
        self.numModalities = len(self.modalities)
        
        self.numSubjects = getattr(X, f"{self.modalities[0]}_data").shape[0]
        #self.numSubjects = X.EEG_data.shape[0]

        self.T = np.array([getattr(X, f"{m}_data").shape[1] for m in self.modalities]) #number of time points               
        #self.T = np.array([X.EEG_data.shape[1], X.MEG_data.shape[1], X.fMRI_data.shape[1]]) #number of time points  

        self.V = getattr(X, f"{self.modalities[0]}_data").shape[2] #number of sources          
        #self.V = X.EEG_data.shape[2] #number of sources

        self.loss_robust = loss_robust
        
        if loss_robust:
            self.epsilon = 1e-3
        else:
            self.epsilon = 1e-6
            
        self.A = 0        

        #C is universal for all subjects/modalities. S(ms) and A(ms) are unique though
        #so we need to create a list for each subject's data for each modality
        self.C = torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((self.V, k), dtype=torch.float))) #softmax upon initialization
        # here Sms has the shape of (m, s, k, V)
        self.Sms = torch.nn.Parameter(torch.nn.Softmax(dim = -2)(torch.rand((self.numModalities, self.numSubjects, k, self.V), dtype = torch.float)))

    
        self.X = [torch.tensor(getattr(X, f"{self.modalities[m]}_data"), dtype = torch.double) for m in range(self.numModalities)]
        #self.X = [torch.tensor(X.EEG_data, dtype = torch.double), torch.tensor(X.MEG_data, dtype = torch.double), torch.tensor(X.fMRI_data, dtype = torch.double)]

        #losses as dict
        self.losses = {"eeg_loss": [], "meg_loss": [], "fmri_loss": []}
        
        #self.eeg_loss = []
        #self.meg_loss = []
        #self.fmri_loss = []
        
    def forward(self):
        #find the unique reconstruction for each modality for each subject
        mle_loss = 0
        for m,modalName in enumerate(self.modalities):

            #X - Xrecon (via MMAA)
            #A = XC
            self.A = self.X[m]@torch.nn.functional.softmax(self.C, dim = 0, dtype = torch.double)
            loss_per_sub = torch.linalg.matrix_norm(self.X[m]-self.A@torch.nn.functional.softmax(self.Sms[m], dim = -2, dtype = torch.double))**2
            
            if self.loss_robust:
                beta  = 3/2 * self.epsilon
                # find the highest number of time points across all modalities               
                max_T = np.max(self.T)
                alpha = 1 + max_T/2  - self.T[m]/2
                mle_loss_m = - (2 * (alpha + 1) + self.T[m])/2 * torch.sum(torch.log(torch.add(loss_per_sub, 2 * beta)))
                mle_loss += mle_loss_m
                
                if torch.sum(loss_per_sub) == 0:
                    print("We hit a 0 loss per sub!")

                self.losses[modalName + "_loss"].append(-mle_loss_m)

                """
                if modalName == "eeg":
                    self.losses["eeg_loss"].append(-mle_loss_m)
                elif modalName == "meg":
                    self.losses["meg_loss"].append(-mle_loss_m)
                elif modalName == "fmri":
                    self.losses["fmri_loss"].append(-mle_loss_m)    
                """              
                    
            else: 
                mle_loss_m = -self.T[m] / 2 * (torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(torch.add(loss_per_sub, self.epsilon)))
                                          - torch.log(torch.tensor(self.T[m])) + 1)
                mle_loss += mle_loss_m
                
                if torch.sum(loss_per_sub) == 0:
                    print("We hit a 0 loss per sub!")
                
                self.losses[modalName + "_loss"].append(-mle_loss_m)
                """
                if modalName == "eeg":
                    self.losses["eeg_loss"].append(-mle_loss_m)
                elif modalName == "meg":
                    self.losses["meg_loss"].append(-mle_loss_m)
                elif modalName == "fmri":
                    self.losses["fmri_loss"].append(-mle_loss_m)    
                """              

        #minimize negative log likelihood
        return -mle_loss


def trainModel(X: Real_Data, numArchetypes=15,seed=32,
              plotDistributions=False,
              learningRate=1e-1,
              numIterations=10000, loss_robust=True,modalities=["eeg", "meg", "fmri"]):
    #seed 
    np.random.seed(seed)
    torch.manual_seed(seed)       
    
    #X = Real_Data(numSubjects=16)
    
    # I dont think this is used:
    ###dim
    #V = X.EEG_data.shape[2]
    #T = np.array([np.shape(X.EEG_data)[1], np.shape(X.MEG_data)[1], np.shape(X.fMRI_data)[1]])
    #k = numArchetypes
    
    model = MMAA(X, numArchetypes, loss_robust, modalities=modalities) 

    #hyperparameters
    lr = learningRate
    n_iter = numIterations

    #loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10) # patience = 10 is default

    # Creating Dataloader object
    loss_Adam = []
    lr_change = []
    tol = 1e-1
    for i in range(n_iter):
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # making a prediction in forward pass
        loss = model.forward()
        # update learning rate
        # scheduler.step(loss)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()

        # store loss into list
        loss_Adam.append(loss.item())

        #print(f"This is the current loss) {loss.item()}")
        #print(f"This is the current iteration {i})")

        #break if loss does not improve
        # this is set to 200 based on prevoius runs
        if i > 200 and np.abs(loss_Adam[-2] - loss_Adam[-1])/np.abs(loss_Adam[-2]) < tol:
            break
        lr_change.append(optimizer.param_groups[0]["lr"])        
    
    #retrieve the matrices and losses we want for plotting
    C = torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double).cpu().detach().numpy()
    Sms = torch.nn.functional.softmax(model.Sms, dim = -2, dtype = torch.double).cpu().detach().numpy()
    #S = np.mean(torch.nn.functional.softmax(model.Sms, dim = -2, dtype = torch.double).cpu().detach().numpy(), axis = 1)

    eeg_loss = model.losses["eeg_loss"]
    meg_loss = model.losses["meg_loss"]
    fmri_loss = model.losses["fmri_loss"]
    
    return C, Sms, eeg_loss, meg_loss, fmri_loss, loss_Adam

if __name__ == "__main__":
    split = 0
    seed = 0
    k = 2 
    iterations = 2
    lossRobust = True
    modalities = ["eeg", "meg"]
    #modalities = ["eeg", "meg", "fmri"]
    
    #X = Real_Data(subjects=range(1, 17), split=split) 
    X = Real_Data(subjects=range(1, 17), split=split)   
        
    C, Sms, eeg_loss, meg_loss, fmri_loss, loss_Adam = trainModel(X, 
                                                                numArchetypes=k,
                                                                seed=seed,
                                                                plotDistributions=False,
                                                                numIterations=iterations,
                                                                loss_robust=lossRobust,
                                                                modalities=modalities)             
    # save the results  
    save_path = f'data/MMAA_results/single_run/{"-".join(modalities)}/split-{split}/k-{k}/seed-{0}/'  
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save C
    np.save(save_path + f'C_split-{split}_k-{k}_seed-{seed}', C)  

    # save all the S matrices
    # filename for sub: S_split-x_k-x_seed-x_sub-x_mod-m
    # filename for average: S_split-x_k-x_seed-x_sub-avg
    #modalities = ['eeg', 'meg', 'fmri']
    m,sub,k,_ = Sms.shape
    for i in range(m):
        for j in range(sub):
            np.save(save_path + f'S_split-{split}_k-{k}_seed-{seed}_sub-{j}_mod-{modalities[i]}', Sms[i,j,:,:])

    S_avg = np.mean(Sms, axis = 1)
    np.save(save_path + f'S_split-{split}_k-{k}_seed-{seed}_sub-avg', S_avg)

    # save all the losses
    # Save the different loss
    # filename: loss_split-x_k-x_seed-x_type-m
    # m will be, eeg,meg,fmri and sum. 
    # sum is the sum of the three losses
    loss = [eeg_loss, meg_loss, fmri_loss,loss_Adam]
    loss_type = ['eeg', 'meg', 'fmri', 'sum']
    for i in range(len(loss)):
        if i == 3:
            np.save(save_path + f'loss_split-{split}_k-{k}_seed-{seed}_type-{loss_type[i]}', np.array(loss[i]))
        else:    
            np.save(save_path + f'loss_split-{split}_k-{k}_seed-{seed}_type-{loss_type[i]}', np.array([int(x.cpu().detach().numpy())for x in loss[i]]))  


  