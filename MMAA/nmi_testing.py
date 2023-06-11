import numpy as np
from nmi import nmi



datapath = f'C:/Users/chwe/Desktop/MMAA_results_first_run/multiple_runs/split_0/'


S1 = np.load(datapath + '/S_matrix_k6_s0_split0.npy')
S2 = np.load(datapath + '/S_matrix_k6_s10_split0.npy')

eeg_nmi = nmi(S1[0], S2[0])
meg_nmi = nmi(S1[1], S2[1])
fmri_nmi = nmi(S1[2], S2[2])

print("end")




"""

def pdd(S1, S2, k1, k2):
            #p(d,d') = ∑_n p(d|n)*p(d'|n)*p(n) (#p(d|n) = s[d,n], p(n) = 1/n)

            # use vectorization to calculate pdd
            # pdd_ = np.sum(S1[k1] * S2[k2]) / S1.shape[1]
            return np.sum(S1[k1] * S2[k2]) / S1.shape[1]

def i(S1, S2):
        #p(d) = ∑_n p(d|n)*p(n), p(d') = ∑_n p(d'|n)*p(n)
        # use vectorization to calculate pd1 and pd2
        pd1 = np.sum(S1, axis=1) / S1.shape[1]
        pd2 = np.sum(S2, axis=1) / S1.shape[1]      
        
        KL = 0
        for k1 in range(S1.shape[0]):
            for k2 in range(S2.shape[0]):
                #kullback-leibler entropy: ∑_(d,d') p(d,d') * log(p(d,d') / (p(d) * p(d')))

                # only calculate pdd once
                pdd_ = pdd(S1, S2, k1, k2)
                KL += pdd_ * np.log(pdd_ / (pd1[k1] * pd2[k2]))     
        return KL 

# the marginal probability of each archetype
pd1 = np.sum(S1, axis=1) / S1.shape[1]
pd2 = np.sum(S2, axis=1) / S1.shape[1] 

# the joint probability of each archetype
pdd_ = 



pdd = pdd(S1, S2, 0, 0)

test = i(S1[0], S1[0])
print("end")
"""
#eeg_nmi = nmi(S1[0], S2[0])
#meg_nmi = nmi(S1[1], S2[1])
#fmri_nmi = nmi(S1[2], S2[2])

