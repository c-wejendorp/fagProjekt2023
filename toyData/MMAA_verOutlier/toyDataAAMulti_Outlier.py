import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcol
from matplotlib.cm import get_cmap
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D

from generate_data_outlier import Synthetic_Data #initializeVoxelss

class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances. See Matplotlib Demos
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines
    
class MMAA(torch.nn.Module):
    def __init__(self, V, T, k, X:Synthetic_Data, numSubjects = 1, numModalities = 1, loss_type = 'normal_mle'): #k is number of archetypes
        super(MMAA, self).__init__()
        
        #For toydataset purposes:
            #k = 10, modalities = 3, subjects = 6, T = 100, V = 5,
        
        #C is universal for all subjects/modalities. S(ms) and A(ms) are unique though
        #so we need to create a list for each subject's data for each modality
        self.C = torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((V, k), dtype=torch.float))) #softmax upon initialization

        # here Sms has the shape of (m, s, k, V)
        self.Sms = torch.nn.Parameter(torch.nn.Softmax(dim = -2)(torch.rand((numModalities, numSubjects, k, V), dtype = torch.float)))

        self.A = 0
        
        if loss_type == 'mle_rob':
            self.epsilon = 1e-3
        else:
            self.epsilon = 1e-6
        
        self.numModalities = numModalities
        self.numSubjects = numSubjects
        self.T = T
        self.V = V
        self.X = [torch.tensor(X.X_eeg, dtype = torch.double), torch.tensor(X.X_meg, dtype = torch.double), torch.tensor(X.X_fmri, dtype = torch.double)]
        
        self.loss_type = loss_type

    def forward(self):
        #vectorize it later
        XCSms = [[0]*self.numSubjects for modality in range(self.numModalities)]
        
        #find the unique reconstruction for each modality for each subject
        loss = 0
        mle_loss_rob = 0
        mle_loss = 0
        for m in range(self.numModalities):

            #X - Xrecon (via MMAA)
            # A = XC
            self.A = self.X[m]@torch.nn.functional.softmax(self.C, dim = 0, dtype = torch.double)
            loss_per_sub = torch.linalg.matrix_norm(self.X[m]-self.A@torch.nn.functional.softmax(self.Sms[m], dim = -2, dtype = torch.double))**2
            
            loss += torch.sum(loss_per_sub)
            mle_loss += -self.T[m] / 2 * (torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(torch.add(loss_per_sub, self.epsilon)))
                                          - torch.log(torch.tensor(self.T[m])) + 1)

            # mle_loss_rob += -self.T[m] / 2 * (torch.log(torch.tensor(2 * torch.pi)) + torch.log(torch.sum(loss_per_sub)/self.T[m] + self.epsilon)) - torch.sum(loss_per_sub)/(2 * (torch.sum(loss_per_sub)/self.T[m] + 1))
            
            beta  = 1/(self.V) * self.epsilon

            alpha = 1 + self.T[2]/2 - self.T[m]/2
            mle_loss_rob_m = - (2 * (alpha + 1) + self.T[m])/2 * torch.sum(torch.log(torch.add(loss_per_sub, 2 * beta)))
            mle_loss_rob += mle_loss_rob_m
            if torch.sum(loss_per_sub) == 0:
                print("Hit it")
            # if mle_loss_rob_m > 0:
            #     print("negative loss???")
                
        if self.loss_type == 'normal_mle': return -mle_loss
        elif self.loss_type == 'mle_rob': return -mle_loss_rob
        elif self.loss_type == 'squared_err': return loss
        else: raise NotImplementedError

    
def toyDataAA(numArchetypes=25,
              loss_type = 'normal_mle',
              numpySeed=32,
              torchSeed=10,
              plotDistributions=True,
              learningRate=1e-1,
              numIterations=10000, 
              T_eeg=100, 
              T_meg=100, 
              T_fmri=500, 
              nr_subjects=10, 
              nr_sources=25, 
              arg_eeg_sources=[np.arange(0,4), np.arange(7,11), np.arange(14,18)], 
              arg_meg_sources=[np.array([0+i*7, 1+i*7, 4+i*7, 5+i*7]) for i in range(3)], 
              arg_fmri_sources=[np.array([1+i*7, 2+i*7, 4+i*7, 6+i*7]) for i in range(3)] + [np.array([24])], 
              activation_timeidx_eeg = np.array([0, 30, 60]), 
              activation_timeidx_meg=np.array([0, 30, 60]) + 10, 
              activation_timeidx_fmri=np.array([0, 30, 60, -10]) + 50):
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

    model = MMAA(V, T, k, X, numModalities=3, numSubjects=nr_subjects, loss_type=loss_type)

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

        #there is 100% a smarter way to do this, but I'm lazy and this is only for toydata
        import matplotlib._color_data as mcd
        palette = list(mcd.XKCD_COLORS.values())[::10] # I'm SORRY FOR THE UGLY COLORS D:
        source_colors = np.array(np.array(palette)[V:])

        arg_eeg_sources_concat = np.concatenate([area for area in arg_eeg_sources])
        arg_meg_sources_concat = np.concatenate([area for area in arg_meg_sources])
        arg_fmri_sources_concat = np.concatenate([area for area in arg_fmri_sources])


        source_eeg_fmri = []
        source_eeg_meg = []
        source_meg_fmri = []
        source_all = []
        for i in range(V):
            if i in arg_eeg_sources_concat and i in arg_meg_sources_concat and i in arg_fmri_sources_concat: 
                source_all.append(i)
            elif i in arg_eeg_sources_concat and i in arg_meg_sources_concat: 
                source_eeg_meg.append(i)
            elif i in arg_eeg_sources_concat and i in arg_fmri_sources_concat:
                source_eeg_fmri.append(i)
            elif i in arg_meg_sources_concat and i in arg_fmri_sources_concat:
                source_meg_fmri.append(i)

        for sub in range(X.nr_subjects):
            if sub == 0 or sub == 2: #AHHHHH TOO MANY PLOTS >:(
                fig, ax = plt.subplots(3)
                for voxel in range(V):
                    for modality in range(3):
                        ax[modality].plot(np.arange(T[modality]), model.X[modality][sub, :, voxel], '-', alpha=1, c=source_colors[voxel]) 
                ax[0].set_title("EEG", fontsize="medium")
                ax[1].set_title("MEG", fontsize="medium")
                ax[2].set_title("fMRI", fontsize="medium")
                fig.suptitle("Plotted Distributions of the Data", fontsize="large")
    

                lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

                # Finally, the legend (that maybe you'll customize differently)
                line = [[(0, 0)]]
                # set up the proxy artist


                lc_eeg_meg = mcol.LineCollection(len(source_eeg_meg) * line, linestyles=['solid' for i in range(len(source_eeg_meg))], colors=source_colors[source_eeg_meg])
                lc_eeg_fmri = mcol.LineCollection(len(source_eeg_fmri) * line, linestyles=['solid' for i in range(len(source_eeg_fmri))], colors=source_colors[source_eeg_fmri])
                lc_meg_fmri = mcol.LineCollection(len(source_meg_fmri) * line, linestyles=['solid' for i in range(len(source_meg_fmri))], colors=source_colors[source_meg_fmri])
                lc_all = mcol.LineCollection(len(source_all) * line, linestyles=['solid' for i in range(len(source_all))], colors=source_colors[source_all])

                # create the legend
                fig.legend([lc_eeg_meg,lc_eeg_fmri,lc_meg_fmri,lc_all], ['EEG+MEG shared', 'EEG+fMRI shared', 'MEG+fMRI shared', 'All shared'], handler_map={type(lc_all): HandlerDashedLines()},
                        handlelength=2.5, handleheight=3, title="Shared source activations", loc = "upper right", bbox_to_anchor=(1.15,1))
                # fig.legend(lines, labels, loc='right')
                fig.tight_layout()
                fig.subplots_adjust()
                plt.savefig(r"toyData\plots\distribution.png", bbox_inches="tight")
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
    # lossCriterion = torch.nn.MSELoss(reduction = "sum")
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
        # scheduler.step(loss)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()

        # store loss into list
        loss_Adam.append(loss.item())
        
        if i > 500 and np.abs(loss_Adam[-2] - loss_Adam[-1])/np.abs(loss_Adam[-2]) < tol:
            break
        lr_change.append(optimizer.param_groups[0]["lr"])

        
    #print("loss list ", loss_Adam) 
    print("final loss: ", loss_Adam[-1])
    if plotDistributions: 
        #plot archetypes
        fig, ax = plt.subplots(4)     

        #plot the different archetypes
        for m in range(3):
            A = np.mean((model.X[m]@torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double)).detach().numpy(), axis = 0)
            for arch in range(k):
                ax[m].plot(range(T[m]), A[:, arch])
        ax[0].set_title("EEG", fontsize="medium")
        ax[1].set_title("MEG", fontsize="medium")
        ax[2].set_title("fMRI", fontsize="medium")
        ax[3].set_title("Plotted Archetype Generator Matrix C")
        ax[-1].plot(range(V), torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double).detach().numpy())
        ax[-1].set_xticks(range(V))
        textstr = '\n'.join((
            'Shared source activations:',
            str(source_all)))
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        fig.text(0.8, 0.95, textstr , fontsize='small',
                verticalalignment='top', bbox=props)
        fig.suptitle("Plotted Archetypes Averaged over Subjects")
        fig.tight_layout()
        plt.savefig(r"toyData\plots\archetypes.png", bbox_inches="tight")
        plt.show()

        ### plot reconstruction
        #m x t x v (averaged over subjects)

        fig, ax = plt.subplots(3)
        for m in range(3):
            A = np.mean((model.X[m]@torch.nn.functional.softmax(model.C, dim = 0, dtype = torch.double)).detach().numpy(), axis = 0)
            Xrecon = A@np.mean(torch.nn.functional.softmax(model.Sms[m], dim = -2, dtype = torch.double).detach().numpy(), axis = 0)
            for voxel in range(V):
                ax[m].plot(np.arange(T[m]), Xrecon[:, voxel], '-', alpha=0.5)

        ax[0].set_title("EEG", fontsize="medium")
        ax[1].set_title("MEG", fontsize="medium")
        ax[2].set_title("fMRI", fontsize="medium")
        fig.suptitle("Reconstructed Distributions after optimizing C and S")
        fig.tight_layout()
        plt.savefig(r"toyData\plots\reconstruction.png")
        plt.show()    

    return loss_Adam

if __name__ == "__main__":

    toyDataAA(numArchetypes=3, torchSeed=10, plotDistributions=True, loss_type='mle_rob', nr_subjects=30)
    toyDataAA(numArchetypes=3, torchSeed=10, plotDistributions=True, loss_type='squared_err', nr_subjects=30)

    