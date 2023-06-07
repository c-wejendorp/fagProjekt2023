import numpy as np
# from toyDataAAMulti import toyDataAA

def nmi(S1, S2):
    def i(S1, S2):
        sources = S1.shape[1]
        
        #p(d|n) = s[d,n], p(d'|n) = s[d',n], p(n) = 1/n
        pdn1 = [S1[:, v] for v in range(sources)]
        pdn2 = [S2[:, v] for v in range(sources)]
        pn = 1 / sources
        
        #p(d,d') = ∑_n p(d|n)*p(d'|n)*p(n)
        pdd = sum([pdn1[v] * pdn2[v] * pn for v in range(sources)])
        
        #p(d) = ∑_n p(d|n)*p(n), p(d') = ∑_n p(d'|n)*p(n)
        pd1 = sum([S1[:, v] * pn for v in range(sources)])
        pd2 = sum([S2[:, v] * pn for v in range(sources)])
        
        #kullback-leibler entropy: ∑_d p(d,d') * log(p(d,d') / (p(d) * p(d')))
        kl_entropy = sum(pdd * np.log(pdd / (pd1 * pd2)))
        
        return kl_entropy
    
    mutual_info = 2 * i(S1, S2) / (i(S1, S1) + i(S2, S2))
    
    assert (mutual_info >= 0 or mutual_info <= 1), f"boundary error. nmi is not between 0 and 1, but got {mutual_info}"
    
    return mutual_info

# if __name__ == "__main__":
#     #Remember to return the S matrix in the toyData code if you want to run this
#     _, S1 = toyDataAA(numArchetypes = 5, numpySeed=2, plotDistributions=False)
#     _, S2 = toyDataAA(numArchetypes = 5, numpySeed=24, plotDistributions=False)
    
#     eeg_nmi = nmi(S1[0], S2[0])
#     meg_nmi = nmi(S1[1], S2[1])
#     fmri_nmi = nmi(S1[2], S2[2])