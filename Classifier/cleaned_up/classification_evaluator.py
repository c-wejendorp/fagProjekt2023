import numpy as np
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os



class Performance_evaluator:
    def __init__(self) -> None:
        pass
    
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

    def bonferoni_p(p_values):
        p_adjusted = multipletests(p_values, method='bonferroni')

        f = open('Classifier/mc_nemar_results.txt', 'a')
        print('______bonf corrected p-values_______', file=f)
        print(p_adjusted, file=f)


    def confusion(y_true, y_predict, arch, pca = True, plot = False):
        #make confusion matrix
        cm = confusion_matrix(y_true, y_predict, labels = ["famous", "scrambled", "unfamiliar"])
        df_cm = pd.DataFrame(cm, index = ["true label: famous", "true label: scrambled", "true label: unfamiliar"],
                            columns = ["predicted: famous", "predicted: scrambled", "predicted: unfamiliar"])

        plt.figure(figsize = (10,7))
        plt.title(f"Confusion matrix for k = {arch}")
        sn.heatmap(df_cm, annot=True)
        if plot:
            plt.show()
        
        path = "Classifier/confusion_plots"
        # make save diractory
        if pca:
            path += "/pca"
        else:
            path += "/no_pca"
        if not os.path.exists(path):
            os.makedirs(path)
            
        plt.savefig(path + f"/k_{arch}")
        


    def plot_results():
        
        
        # calculate the mean and std for each number of archetypes 
        LR_pca_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_pca_loss.items()], dtype="float64")
        LR_pca_std = np.array([np.std(loss) for archetype, loss in LR_pca_loss.items()])
        
        LR_mean = np.array([[archetype, np.mean(loss)] for archetype, loss in LR_loss.items()], dtype="float64")
        LR_std = np.array([np.std(loss) for archetype, loss in LR_loss.items()])
        
        #plot the mean values with std
        plt.errorbar(LR_pca_mean[:,0], LR_pca_mean[:,1], yerr = LR_pca_std, label = "LR_pca")
        plt.errorbar(LR_mean[:,0], LR_mean[:,1], yerr = LR_std, label = "LR")
        plt.legend()
        # make the x ticks integers 
        plt.xticks(LR_pca_mean[:,0])
        plt.title("Final classification accuracy for different number of archetypes training data")
        plt.xlabel("Number of archetypes")
        plt.ylabel("Final loss")
        # plt.annotate()
        plt.savefig(savepath + f"class_error__{'-'.join(modalityComb)}.txt.png")    
        #plt.show()  