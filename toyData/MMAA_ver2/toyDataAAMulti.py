import torch 
import numpy as np
import matplotlib.pyplot as plt
from generate_data import Synthetic_Data #initializeVoxelss
    
class MMAA(torch.nn.Module):
    def __init__(self, V, T, k, X:Synthetic_Data, numSubjects = 1, numModalities = 1): #k is number of archetypes
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
        self.X = [torch.tensor(X.X_eeg, dtype = torch.double), torch.tensor(X.X_meg, dtype = torch.double), torch.tensor(X.X_fmri, dtype = torch.double)]

    def forward(self):
        #vectorize it later
        XCSms = [[0]*self.numSubjects for modality in range(self.numModalities)]
        
        #find the unique reconstruction for each modality for each subject
        loss = 0
        for m in range(self.numModalities):

            #X - Xrecon (via MMAA)
            # A = XC
            self.A = self.X[m]@torch.nn.functional.softmax(self.C, dim = 0, dtype = torch.double)
            loss_per_sub = torch.linalg.matrix_norm(self.X[m]-self.A@torch.nn.functional.softmax(self.Sms[m], dim = -2, dtype = torch.double))**2
            
            loss += torch.sum(loss_per_sub)
            
        #XCSms is a list of list of tensors. Here we convert everything to tensors
        # XCSms = torch.stack([torch.stack(XCSms[i]) for i in range(len(XCSms))])

        # # i think we also need to save the reconstruction
        # self.XCSms = XCSms

        return loss

    
def toyDataAA(numArchetypes=25,
              numpySeed=32,
              torchSeed=0,
              plotDistributions=False,
              learningRate=1e-1,
              numIterations=10000, 
              T_eeg=100, 
              T_meg=100, 
              T_fmri=500, 
              nr_subjects=10, 
              nr_sources=25, 
              arg_eeg_sources=(np.arange(0,4), np.arange(7,11), np.arange(14,18)), 
              arg_meg_sources=(np.array([0+i*7, 1+i*7, 4+i*7, 5+i*7]) for i in range(3)), 
              arg_fmri_sources=(np.array([1+i*7, 2+i*7, 4+i*7, 6+i*7]) for i in range(3)), 
              activation_timeidx_eeg = np.array([0, 30, 60]), 
              activation_timeidx_meg=np.array([0, 30, 60]) + 10, 
              activation_timeidx_fmri=np.array([0, 30, 60]) + 50):
    #seed 
    np.random.seed(numpySeed)
    torch.manual_seed(torchSeed)
    
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

    X = Synthetic_Data(T_eeg=T_eeg, 
                       T_meg=T_meg, 
                       T_fmri=T_fmri, 
                       nr_subjects=nr_subjects, 
                       nr_sources=nr_sources, 
                       arg_eeg_sources=arg_eeg_sources, 
                       arg_meg_sources=arg_meg_sources, 
                       arg_fmri_sources=arg_fmri_sources, 
                       activation_timeidx_eeg = activation_timeidx_eeg, 
                       activation_timeidx_meg=activation_timeidx_meg, 
                       activation_timeidx_fmri=activation_timeidx_fmri)
    
    ###dim
    V = X.nr_sources
    T = np.array([np.shape(X.X_eeg)[1], np.shape(X.X_meg)[1], np.shape(X.X_fmri)[1]])
    k = numArchetypes

    model = MMAA(V, T, k, X, numModalities=3, numSubjects=nr_subjects)

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
        for sub in range(X.nr_subjects):
            if sub == 0: #AHHHHH TOO MANY PLOTS >:(
                _, ax = plt.subplots(3)
                for voxel in range(V):
                    for modality in range(3):
                        ax[modality].plot(np.arange(T[modality]), model.X[modality][sub, :, voxel], '-', alpha=0.5) 
                plt.savefig(r"C:\University\fagProjekt2023\toyData\plots\distribution")
                plt.show()
            

    ###create X matrix dependent on modality and subject
    # modality x subject x time x voxel
    #Xms = np.zeros((3, numSubjects, T, V))
    #
    #mod_list = [meg, eeg, fmri]
    #for idx_modality, data in enumerate(mod_list):        
    #    Xms[idx_modality, :, :, :] = data #This works but if time: just concanate it all along some axis

    #Xms = torch.tensor(Xms, dtype = torch.double)


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
    plt.savefig(r"C:\University\fagProjekt2023\toyData\plots\archetypes")
    plt.show()
    
    ### plot reconstruction
    #m x t x v (averaged over subjects)

    _, ax = plt.subplots(3)
    for m in range(3):
        A = np.mean((model.X[m]@torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double)).detach().numpy(), axis = 0)
        for voxel in range(V):
            Xrecon = A@np.mean(torch.nn.functional.softmax(model.Sms[m], dim = -2, dtype = torch.double).detach().numpy(), axis = 0)
            ax[m].plot(np.arange(T[m]), Xrecon[:, voxel], '-', alpha=0.5)

    
    plt.savefig(r"C:\University\fagProjekt2023\toyData\plots\reconstruction")
    plt.show()    
    
    #return data,archeTypes,loss_Adam

if __name__ == "__main__":
    toyDataAA(plotDistributions=True)
    