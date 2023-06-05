import torch 
import numpy as np
import matplotlib.pyplot as plt
from loadData import Real_Data
    
class MMAA(torch.nn.Module):
    def __init__(self, X: Real_Data, k : int,numModalities=3):
    #def __init__(self, V, T, k, X: Real_Data, numSubjects = 1, numModalities = 1): #k is number of archetypes
        super(MMAA, self).__init__()

        self.numModalities = numModalities
        self.numSubjects = X.EEG_data.shape[0]
        self.T = np.array([X.EEG_data.shape[1], X.MEG_data.shape[1], X.fMRI_data.shape[1]]) #number of time points      
        self.V = X.EEG_data.shape[2] #number of sources
        self.epsilon = 1e-6

        self.A = 0        

        #C is universal for all subjects/modalities. S(ms) and A(ms) are unique though
        #so we need to create a list for each subject's data for each modality
        self.C = torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((self.V, k), dtype=torch.float))) #softmax upon initialization
        # here Sms has the shape of (m, s, k, V)
        self.Sms = torch.nn.Parameter(torch.nn.Softmax(dim = -2)(torch.rand((self.numModalities, self.numSubjects, k, self.V), dtype = torch.float)))

        self.X = [torch.tensor(X.EEG_data, dtype = torch.double), torch.tensor(X.MEG_data, dtype = torch.double), torch.tensor(X.fMRI_data, dtype = torch.double)]
        # test=2
        
    # def forward(self):
    #     #vectorize it later
    #     XCSms = [[0]*self.numSubjects for modality in range(self.numModalities)]
        
    #     #find the unique reconstruction for each modality for each subject
    #     loss = 0
    #     for m in range(self.numModalities):

    #         #X - Xrecon (via MMAA)
    #         # A = XC
    #         self.A = self.X[m]@torch.nn.functional.softmax(self.C, dim = 0, dtype = torch.double)
    #         loss_per_sub = torch.linalg.matrix_norm(self.X[m]-self.A@torch.nn.functional.softmax(self.Sms[m], dim = -2, dtype = torch.double))**2
            
    #         loss += torch.sum(loss_per_sub)
            
    #     #XCSms is a list of list of tensors. Here we convert everything to tensors
    #     # XCSms = torch.stack([torch.stack(XCSms[i]) for i in range(len(XCSms))])

    #     # # i think we also need to save the reconstruction
    #     # self.XCSms = XCSms

        
    #     return loss   

    def forward(self):
        #find the unique reconstruction for each modality for each subject
        loss = 0
        mle_loss = 0
        for m in range(self.numModalities):

            #X - Xrecon (via MMAA)
            #A = XC
            self.A = self.X[m]@torch.nn.functional.softmax(self.C, dim = 0, dtype = torch.double)
            loss_per_sub = torch.linalg.matrix_norm(self.X[m]-self.A@torch.nn.functional.softmax(self.Sms[m], dim = -2, dtype = torch.double))**2
            
            loss += torch.sum(loss_per_sub)
            mle_loss += -self.T[m] / 2 * (torch.log(torch.tensor(2 * torch.pi)) + torch.log(torch.sum(loss_per_sub) + self.epsilon) 
                                          - torch.log(torch.tensor(self.T[m])) + 1)
            if torch.sum(loss_per_sub) == 0:
                print("Hit it")

        #minimize negative log likelihood
        #return loss
        return -mle_loss


def trainModel(numArchetypes=25,
              numpySeed=32,
              torchSeed=0,
              plotDistributions=False,
              learningRate=1e-1,
              numIterations=10000):
    #seed 
    np.random.seed(numpySeed)
    torch.manual_seed(torchSeed)
    path = "/Users/helenakeitum/Desktop"    
    
    X = Real_Data(2)
    
    ###dim
    V = X.EEG_data.shape[2]
    T = np.array([np.shape(X.EEG_data)[1], np.shape(X.MEG_data)[1], np.shape(X.fMRI_data)[1]])
    k = numArchetypes
    
    model = MMAA(X, k, numModalities=3)    
    
    if plotDistributions:        
        for sub in range(X.EEG_data.shape[0]): #num of subjects
            if sub == 0: #AHHHHH TOO MANY PLOTS >:(
                _, ax = plt.subplots(3)
                for voxel in range(V):
                    for modality in range(3):
                        ax[modality].plot(np.arange(T[modality]), model.X[modality][sub, :, voxel], '-', alpha=0.5) 
                plt.savefig(path)
                plt.show()
            

    
    #hyperparameters
    lr = learningRate
    n_iter = numIterations

    #loss function
    lossCriterion = torch.nn.MSELoss(reduction = "sum")
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10) # patience = 10 is default

    # Creating Dataloader object
    loss_Adam = []
    lr_change = []
    tol = 1e-6
    for i in range(n_iter):
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # making a prediction in forward pass
        loss = model.forward()
        # update learning rate
        scheduler.step(loss)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()

        # store loss into list
        loss_Adam.append(loss.item())

        #print(f"This is the current loss) {loss.item()}")
        #print(f"This is the current iteration {i})")

        
        if i > 500 and np.abs(loss_Adam[-2] - loss_Adam[-1]) < tol:
            break
        lr_change.append(optimizer.param_groups[0]["lr"])

        
    #print("loss list ", loss_Adam) 
    print("final loss: ", loss_Adam[-1])
    
    #plot archetypes
    _, ax = plt.subplots(4)     

    #plot the different archetypes
    for m in range(3):
        A = np.mean((model.X[m]@torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double)).detach().numpy(), axis = 0)
        for arch in range(k):
            ax[m].plot(range(T[m]), A[:, arch])
    ax[-1].plot(range(V), torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double).detach().numpy())
    plt.savefig(path)
    plt.show()
    
    ### plot reconstruction
    #m x t x v (averaged over subjects)

    _, ax = plt.subplots(3)
    for m in range(3):
        A = np.mean((model.X[m]@torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double)).detach().numpy(), axis = 0)
        for voxel in range(V):
            Xrecon = A@np.mean(torch.nn.functional.softmax(model.Sms[m], dim = -2, dtype = torch.double).detach().numpy(), axis = 0)
            ax[m].plot(np.arange(T[m]), Xrecon[:, voxel], '-', alpha=0.5)

    
    plt.savefig(path)
    plt.show()    
    
    #return data,archeTypes,loss_Adam

if __name__ == "__main__":
    trainModel(plotDistributions=False,numIterations=100)