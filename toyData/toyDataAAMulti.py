import torch 
import numpy as np
import matplotlib.pyplot as plt
    
class MMAA(torch.nn.Module):
    def __init__(self, V, T, k, Xms, numSubjects = 1, numModalities = 1): #k is number of archetypes
        super(MMAA, self).__init__()
        
        #For toydataset purposes:
            #k = 10, modalities = 3, subjects = 6, T = 100, V = 5,
        
        #C is universal for all subjects/modalities. S(ms) and A(ms) are unique though
        #so we need to create a list for each subject's data for each modality
        self.C = torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((V, k), dtype=torch.double))) #softmax upon initialization

        # here Sms has the shape of (numModalities, numSubjects, k, V)
        self.Sms = torch.nn.Parameter(torch.nn.Parameter(torch.rand(numModalities, numSubjects, T, )))
        # as a parameterlist 
        #remove list of list since T is not in here
        
        #should maybe change above to something called PARAMETERLIST 
        #self.Sms = torch.nn.ParameterList([[torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((k, V), dtype=torch.double)))]*numSubjects for m in range(numModalities)])

        self.A = 0
        self.Xms = Xms
        
        self.numModalities = numModalities
        self.numSubjects = numSubjects
        self.T = T
        self.V = V
    
    def soft_fun(self, M):
        """Implements softmax along columns to respect matrix constraints"""
        
        softmax = torch.nn.Softmax(dim = 0)
        softM = softmax(M)
        return softM

    def forward(self, X):
        #vectorize it later
        XCSms = [[0]*self.numSubjects for modality in range(self.numModalities)]
        
        #find the unique reconstruction for each modality for each subject
        #loss = 0
        for m in range(self.numModalities):
            XC = torch.matmul(self.Xms[m], self.soft_fun(self.C))
            self.A = XC
            XCS = torch.matmul(XC, self.soft_fun(self.Sms[m]))
            XCSms[m] = XCS
            #update loss as Xms - XCms
        
        #XCSms is a list of list of tensors. Here we convert everything to tensors
        XCSms = torch.stack([torch.stack(XCSms[i]) for i in range(len(XCSms))])

        # i think we also need to save the reconstruction
        self.XCSms = XCSms

        return loss
    
    
def toyDataAA(numVoxels=5,timeSteps=100,numArchetypes=10,numpySeed=32,torchSeed=0,plotDistributions=False,learningRate=1e-3,numIterations=10000, numSubjects=6):
    #seed 
    np.random.seed(numpySeed)
    torch.manual_seed(torchSeed)

    ###dim
    V = numVoxels
    T = timeSteps
    k = numArchetypes
    
    #voxel distribution follows certain archetypes
    vox1, vox2 = np.random.normal(0.25, 0.01, size = T), np.random.normal(0.25, 0.01, size = T)
    vox3 = np.random.normal(0.5, 0.01, size = T)
    vox4, vox5 = np.random.normal(0.75, 0.01, size = T), np.random.normal(0.75, 0.01, size = T)

    voxels = [vox1, vox2, vox3, vox4, vox5]
    
    ###initialize the a three-dimensional array for each modality (subject, time, voxel)
    meg = np.array([np.array([[voxels[v][t] for v in range(numVoxels)] for t in range(T)]) for _ in range(numSubjects)]) 
    eeg = np.array([np.array([[voxels[v][t] for v in range(numVoxels)] for t in range(T)]) for _ in range(numSubjects)]) 
    fmri = np.array([np.array([[voxels[v][t] for v in range(numVoxels)] for t in range(T)]) for _ in range(numSubjects)]) 
    
    if plotDistributions:        
        for sub in range(meg.shape[0]):
            _, ax = plt.subplots(3)
            for voxel in range(V):
                ax[0].plot(np.arange(T), meg[sub, :, voxel], '-', alpha=0.5)
                ax[1].plot(np.arange(T), eeg[sub, :, voxel], '-', alpha=0.5)
                ax[2].plot(np.arange(T), fmri[sub, :, voxel], '-', alpha=0.5)
            plt.show()

    ###create X matrix dependent on modality and subject
    # modality x subject x time x voxel
    Xms = np.zeros((3, numSubjects, T, V))
    
    mod_list = [meg, eeg, fmri]
    for idx_modality, data in enumerate(mod_list):        
        Xms[idx_modality, :, :, :] = data #This works but if time: just concanate it all along some axis

    Xms = torch.tensor(Xms)

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
        Xrecon = model.forward()
        # calculating the loss between original and predicted data points
        loss = lossCriterion(Xrecon, Xms)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()
        # store loss into list
        loss_Adam.append(loss.item())

    print("loss list ", loss_Adam) 
    print("final loss: ", loss_Adam[-1])
         

    #return data,archeTypes,loss_Adam

if __name__ == "__main__":
    toyDataAA(numIterations=5000, numArchetypes=3)
    