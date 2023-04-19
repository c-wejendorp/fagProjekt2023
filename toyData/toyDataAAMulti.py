import torch 
import numpy as np
import matplotlib.pyplot as plt
from generate_data import initializeVoxels
    
class MMAA(torch.nn.Module):
    def __init__(self, V, T, k, Xms, numSubjects = 1, numModalities = 1): #k is number of archetypes
        super(MMAA, self).__init__()
        
        #For toydataset purposes:
            #k = 10, modalities = 3, subjects = 6, T = 100, V = 5,
        
        #C is universal for all subjects/modalities. S(ms) and A(ms) are unique though
        #so we need to create a list for each subject's data for each modality
        self.C = torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((V, k), dtype=torch.float))) #softmax upon initialization

        # here Sms has the shape of (m, s, k, V)
        self.Sms = torch.nn.Parameter(torch.nn.Softmax(dim = -2)(torch.rand((numModalities, numSubjects, k, V), dtype = torch.float)))

        self.A = 0
        
        self.numModalities = numModalities
        self.numSubjects = numSubjects
        self.T = T
        self.V = V

    def forward(self, X):
        #vectorize it later
        XCSms = [[0]*self.numSubjects for modality in range(self.numModalities)]
        
        #find the unique reconstruction for each modality for each subject
        loss = 0
        for m in range(self.numModalities):            
            #X - Xrecon (via MMAA)
            # A = XC
            self.A = X[m]@torch.nn.functional.softmax(self.C, dim = 0, dtype = torch.double)
            loss_per_sub = torch.linalg.matrix_norm(X[m]-self.A@torch.nn.functional.softmax(self.Sms[m], dim = -2, dtype = torch.double))**2
            loss += torch.sum(loss_per_sub)
            
        #XCSms is a list of list of tensors. Here we convert everything to tensors
        # XCSms = torch.stack([torch.stack(XCSms[i]) for i in range(len(XCSms))])

        # # i think we also need to save the reconstruction
        # self.XCSms = XCSms

        return loss
    
    
def toyDataAA(numVoxels=5,timeSteps=100,numArchetypes=10,numpySeed=32,torchSeed=0,plotDistributions=False,learningRate=1e-3,numIterations=10000, numSubjects=6):
    #seed 
    np.random.seed(numpySeed)
    torch.manual_seed(torchSeed)

    ###dim
    V = numVoxels
    T = timeSteps
    k = numArchetypes
    
    #activation times
    # initialize voxels

    #def initializeVoxels(V, T, means):
    #    """initializes however many voxels we want"""
    #    #initialize "empty" voxels
    #    voxels = []
    #    for i in range(numVoxels):
    #        voxels.append(np.zeros(T))        
#
    #    timestamps = np.array_split(list(range(T)), V)
    #    for i in range(len(voxels)): 
    #        voxels[i][timestamps[i]] = np.random.normal(means[i], 0.01, size = len(timestamps[i]))
    #    
    #    return voxels
    #
    #
    # voxels = initializeVoxels(V, T, [0.1, 0.5, 0.9])

    X = initializeVoxels(V=V, T = T, s = numSubjects, m = 3, eeg_only=6, meg_only=6, fmri_only=6, eeg_meg_shared=3, eeg_fmri_shared=3, meg_fmri_shared=3, golden_voxel=3)
    
    ###initialize the a three-dimensional array for each modality (subject, time, voxel)
    #meg = np.array([np.array([[voxels[v][t] for v in range(numVoxels)] for t in range(T)]) for _ in range(numSubjects)]) 
    #eeg = np.array([np.array([[voxels[v][t] for v in range(numVoxels)] for t in range(T)]) for _ in range(numSubjects)]) 
    #fmri = np.array([np.array([[voxels[v][t] for v in range(numVoxels)] for t in range(T)]) for _ in range(numSubjects)]) 
    
    #if plotDistributions:        
    #    for sub in range(meg.shape[0]):
    #        _, ax = plt.subplots(3)
    #        for voxel in range(V):
    #            ax[0].plot(np.arange(T), meg[sub, :, voxel], '-', alpha=0.5)
    #            ax[1].plot(np.arange(T), eeg[sub, :, voxel], '-', alpha=0.5)
    #            ax[2].plot(np.arange(T), fmri[sub, :, voxel], '-', alpha=0.5)
    #        plt.show()
    
    if plotDistributions:        
        for sub in range(X.shape[1]):
            _, ax = plt.subplots(3)
            for voxel in range(V):
                ax[0].plot(np.arange(T), X[0,sub, :, voxel], '-', alpha=0.5)
                ax[1].plot(np.arange(T), X[1,sub, :, voxel], '-', alpha=0.5)
                ax[2].plot(np.arange(T), X[2,sub, :, voxel], '-', alpha=0.5)
            plt.show()
            

    ###create X matrix dependent on modality and subject
    # modality x subject x time x voxel
    #Xms = np.zeros((3, numSubjects, T, V))
    #
    #mod_list = [meg, eeg, fmri]
    #for idx_modality, data in enumerate(mod_list):        
    #    Xms[idx_modality, :, :, :] = data #This works but if time: just concanate it all along some axis

    #Xms = torch.tensor(Xms, dtype = torch.double)

    Xms = torch.tensor(X, dtype = torch.double)

    #hyperparameters
    lr = learningRate
    n_iter = numIterations

    #loss function
    model = MMAA(V, T, k, Xms, numModalities=3, numSubjects=numSubjects)
    lossCriterion = torch.nn.MSELoss(reduction = "sum")
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Creating Dataloader object
    loss_Adam = []
    for i in range(n_iter):
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # making a prediction in forward pass
        loss = model.forward(Xms)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()
        # store loss into list
        loss_Adam.append(loss.item())

    #print("loss list ", loss_Adam) 
    print("final loss: ", loss_Adam[-1])
    
    #plot archetypes
    _, ax = plt.subplots(3)     
    Am = np.mean((Xms@torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double)).detach().numpy(), axis = 1)
    #plot the different archetypes
    for m in range(3):
        for arch in range(k):
            ax[m].plot(range(T), Am[m, :, arch])
    plt.show()
    print(2)
         
    #return data,archeTypes,loss_Adam

if __name__ == "__main__":
    toyDataAA(numIterations=3000, numSubjects=1, numArchetypes=3, plotDistributions=True, numVoxels=30)
    