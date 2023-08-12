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
    """class that performs the MM(M)AA"""
    
    def __init__(self, X: Real_Data, k : int, loss_robust: bool, modalities = ["eeg", "meg", "fmri"], numCondition = 3):
        """
        X (Real_Data obj): Dataset with dimension [m, s, (c), T, V(*3)] (c) when "multicondition" and (*3) when "spatial",
        k (int): number of archetypes,
        loss_robust (bool): if True, the loss will use a log maximum a posteriori. If False, will use Euclidean distance,
        modalities (list): list of modalities used i.e eeg, meg and fmri,
        numCondition (int): defaults to 3 for "famous", "unfamiliar" and "scrambled" face types
        """
        
        super(MMAA, self).__init__()
        
        self.numCondition = numCondition
        self.modalities = modalities
        self.numModalities = len(self.modalities)
        self.numSubjects = getattr(X, f"{self.modalities[0]}_data").shape[0]
        self.concatenation_type = X.concatenation_type
        
        # number of time points    
        self.T = np.array([getattr(X, f"{m}_data").shape[1] for m in self.modalities])            
        
        # number of sources  
        self.V = getattr(X, f"{self.modalities[0]}_data").shape[-1]         
        
        self.loss_robust = loss_robust
        
        # assign threshold value
        if loss_robust:
            self.epsilon = 1e-3
        else:
            self.epsilon = 1e-6
        
        # archetype matrix placeholder
        self.A = 0        

        # archetype generator matrix
        self.C = torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((self.V, k), dtype=torch.float))) #softmax upon initialization
        
        if self.concatenation_type == "multicondition":
            # here Sms has the shape of (m, s, c, k, V)
            self.Sms = torch.nn.Parameter(torch.nn.Softmax(dim = -2)(torch.rand((self.numModalities, 
                                                                                 self.numSubjects, 
                                                                                 self.numCondition, 
                                                                                 k, self.V), dtype = torch.float)))
        elif self.concatenation_type == "spatial":
            # here Sms has the shape of (m, s, k, V*3)
            self.Sms = torch.nn.Parameter(torch.nn.Softmax(dim = -2)(torch.rand((self.numModalities, 
                                                                                 self.numSubjects, k, self.V), 
                                                                                 dtype = torch.float)))
        else:
            raise ValueError("only multicondition or spatial are valid arguments")

        # create full dataset [m, s, (c), T, V(*3)]
        self.X = [torch.tensor(getattr(X, f"{self.modalities[m]}_data"), dtype = torch.double) for m in range(self.numModalities)]
    
        #losses as dict
        self.losses = {"eeg_loss": [], "meg_loss": [], "fmri_loss": []}
        
    def forward(self):
        """computes the loss for each modality and subject by loss_method(X - XCS)"""
        
        mle_loss = 0
        for m,modalName in enumerate(self.modalities):
            #A = XC
            self.A = self.X[m]@torch.nn.functional.softmax(self.C, dim = 0, dtype = torch.double)
            #X - Xrecon (via MMAA)
            if self.concatenation_type == "spatial":
                loss_per_sub = torch.linalg.matrix_norm(self.X[m]-self.A@torch.nn.functional.softmax(self.Sms[m], dim = -2, dtype = torch.double))**2
            elif self.concatenation_type == "multicondition":
                loss_per_sub = torch.linalg.matrix_norm(torch.sum(self.X[m]-self.A@torch.nn.functional.softmax(self.Sms[m], dim = -2, dtype = torch.double),axis=1))**2
            
            if self.loss_robust:
                #beta value for gamma distribution loss
                beta  = 3/2 * self.epsilon
                
                # find the highest number of time points across all modalities               
                max_T = np.max(self.T)
                
                #make sure first factor of the final loss is consistent across all modalities
                alpha = 1 + max_T/2  - self.T[m]/2
                
                #compute loss
                mle_loss_m = - (2 * (alpha + 1) + self.T[m])/2 * torch.sum(torch.log(torch.add(loss_per_sub, 2 * beta)))
                    
            else: 
                # TODO: change this some time after some pondering of what it should do :))
                mle_loss_m = -self.T[m] / 2 * (torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(torch.add(loss_per_sub, self.epsilon)))
                                          - torch.log(torch.tensor(self.T[m])) + 1)
            
            mle_loss += mle_loss_m    

            self.losses[modalName + "_loss"].append(-mle_loss_m)         

        #minimize negative log likelihood
        return -mle_loss


def trainModel(X: Real_Data, numArchetypes = 15,seed = 32,
              learningRate = 1e-1,
              numIterations = 10000, loss_robust = True, modalities = ["eeg", "meg", "fmri"]):
    """
    X (Real_Data obj): dataset with dimension [m, s, (c), T, V(*3)] (c) when "multicondition" and (*3) when "spatial",
    numArchetypes (int): the number of archetypes to optimize for in the MM(M)AA,
    seed (int): ensure reproducible results,
    learningRate: learning rate used for the optimizer,
    numIterations: max number of steps for the optimizing procedure before termination,
    loss_robust (bool): if True, uses log maximum a posteriori. False uses Eucledian,
    modalities (list): which modalities to extract from the data X
    """
    
    #seed 
    np.random.seed(seed)
    torch.manual_seed(seed)       
    
    model = MMAA(X, numArchetypes, loss_robust, modalities=modalities) 

    #hyperparameters
    lr = learningRate
    n_iter = numIterations

    #loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # Creating Dataloader object
    loss_Adam = []
    lr_change = []
    tol = 1e-1
    for i in range(n_iter):
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        
        # making a prediction in forward pass
        loss = model.forward()
        
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        
        # updating the parameters after each iteration
        optimizer.step()

        # store loss into list
        loss_Adam.append(loss.item())

        #break if loss does not improve
        # this is set to 200 based on previous runs
        if i > 200 and np.abs(loss_Adam[-2] - loss_Adam[-1])/np.abs(loss_Adam[-2]) < tol:
            break
        lr_change.append(optimizer.param_groups[0]["lr"])        
    
    # retrieve the matrices and losses we want for plotting
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
    iterations = 10
    lossRobust = True
    modalities = ["eeg", "meg", "fmri"]
    concatenation_type = "multicondition"
    
    X = Real_Data(subjects=range(10, 16), concatenation_type="multicondition", split = 0)   
        
    C, Sms, eeg_loss, meg_loss, fmri_loss, loss_Adam = trainModel(X, 
                                                                numArchetypes=k,
                                                                seed=seed,
                                                                plotDistributions=False,
                                                                numIterations=iterations,
                                                                loss_robust=lossRobust,
                                                                modalities=modalities, 
                                                                concatenation_type=concatenation_type)             
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


  