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