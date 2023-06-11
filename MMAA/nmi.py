import numpy as np
# from toyDataAAMulti import toyDataAA

def nmi(S1, S2):
    def i(S1, S2):
        def pdd(S1, S2, k1, k2):
            #p(d,d') = ∑_n p(d|n)*p(d'|n)*p(n) (#p(d|n) = s[d,n], p(n) = 1/n)
            # moving the 1/n to the end of the sum      
            # 
            # 
            return sum([S1[k1][v] * S2[k2][v] * 1 / S1.shape[1] for v in range(S1.shape[1])])      
        
            return np.array(sum([S1[k1][v] * S2[k2][v] for v in range(S1.shape[1])])) * 1 / S1.shape[1]            
        
        #p(d) = ∑_n p(d|n)*p(n), p(d') = ∑_n p(d'|n)*p(n)
        # moving the 1/n to the end of the sum
        pd1 = np.array(sum([S1[:, v] for v in range(S1.shape[1])])) * 1 / S1.shape[1]
        pd2 = np.array(sum([S2[:, v] for v in range(S1.shape[1])])) * 1 / S1.shape[1]        
        
        kl = 0
        for k1 in range(S1.shape[0]):
            for k2 in range(S2.shape[0]):
                #kullback-leibler entropy: ∑_(d,d') p(d,d') * log(p(d,d') / (p(d) * p(d')))

                # only calculate pdd once
                pdd_ = pdd(S1, S2, k1, k2)
                k1 += pdd_ * np.log(pdd_ / (pd1[k1] * pd2[k2]))               
        
        return kl
    
    mutual_info = 2 * i(S1, S2) / (i(S1, S1) + i(S2, S2))
    
    assert (mutual_info >= 0 and mutual_info <= 1), f"boundary error. nmi is not between 0 and 1, but got {mutual_info}"
    
    return mutual_info

# if __name__ == "__main__":
#     #Remember to return the S matrix in the toyData code if you want to run this
#     _, S1 = toyDataAA(numArchetypes = 5, numpySeed=2, plotDistributions=False)
#     _, S2 = toyDataAA(numArchetypes = 5, numpySeed=24, plotDistributions=False)
    
#     eeg_nmi = nmi(S1[0], S2[0])
#     meg_nmi = nmi(S1[1], S2[1])
#     fmri_nmi = nmi(S1[2], S2[2])