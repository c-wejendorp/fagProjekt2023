import scipy.stats
import numpy as np
from KNN_classifier import train_KNN
from Multinomial_log_reg import train_LR
import os 
def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat)
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    # print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    # print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    # print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p

"""
TODO: compute the yhats of the best models etc. 
A good idea is to save the predictions and true values somewhere
so there's no need to rerun any of the 

Also implement a baseline that does whatever it wants :)
"""

if __name__ == '__main__':
    # NOTE: I did not have time to test this code 
    
    # path to save the yhats
    output_path = "Classifier/results/"
    archetype = 8
    seed = 10
    _, y_hat_KNN_example, y_true_KNN = train_KNN(K_neighbors=10,distance_measure='Euclidean',pca_data=True, multi=True, archetypes=archetype, seed=seed)
    _, y_hat_LR_example, y_true_LR = train_LR(pca_data=True, multi=True, archetypes= archetype, seed=seed)
    
    assert y_true_KNN == y_true_LR
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    thetahat, CI, p = mcnemar(y_true_KNN, y_hat_KNN_example, y_hat_LR_example, alpha=0.05)
    

    
    
    
    