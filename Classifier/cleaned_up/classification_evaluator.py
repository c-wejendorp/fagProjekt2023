import numpy as np
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os
import json
from collections import defaultdict

CLASSIFIERS_W_PARAMS = ["LR_reg_pacc"]

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
        

    @staticmethod
    def plot_results(raw_result_path, concatenate_seed_results:bool):
        """Plots all mean classifier results with their standard deviation over 
        different archetypes. 
        
        The function is generalized such that there can be added other classifiers/different input data
        classifiers without problem, if the data is saved with the same structure as the existing ones.

        Args:
            raw_result_path (str): Path to the directory where the raw classifier results were written to
            concatenate_seed_results (bool): True if the seed results should be concatenated

        Raises:
            NotImplementedError: If concatenate_seed_results is False, as this has not been implemented
        """
        
        # Load in the classification results
        results = {}
        for res_file in os.listdir(raw_result_path):
            if not ("result" in res_file and ".json" in res_file): continue #skip the checkpoint files and extra junk
            
            with open(os.path.join(raw_result_path, res_file)) as f:
                archetype_result = json.load(f) 
                
            for key in archetype_result.keys():
                if key in results:
                    results[key].update(archetype_result[key])
                else:
                    results[key] = archetype_result[key]

        if concatenate_seed_results: 
            classifier_params = defaultdict(lambda: {})
            
            # concatenate all seed results
            for classifier, classifier_result in results.items():
                for archetype, arch_result in classifier_result.items():
                    if classifier in CLASSIFIERS_W_PARAMS:
                        # If the classifier has different parameter results find the param with highest mean acc
                        #through the seeds 
                        
                        param_accs = {}
                        # Concatenate seed results
                        for param, seed_acc in arch_result.items():
                            param_accs[param] = sum(list(seed_acc.values()), [])
                        
                        # get best classifier parameter
                        param_means = {param: np.mean(accs) for param,accs in param_accs.items()}
                        best_param = max(param_means, key=param_means.get)
                        
                        # overwrite the archetype result using that classifier with the results of the best classifier parameter for mean acc
                        results[classifier][archetype] = param_accs[best_param]
                        
                        # note down the best parameter
                        classifier_params[classifier][archetype] = best_param

                    else:
                        results[classifier][archetype] = sum(list(arch_result.values()), [])

            # Plot the classifier results
            for classifier in results.keys():
                # Skip empty classifier results
                if not results[classifier]: continue 
                
                arch_means = np.array([[archetype, np.mean(loss)] for archetype, loss in results[classifier].items()], dtype="float64")
                stds = np.array([np.std(loss) for archetype, loss in results[classifier].items()])
                plt.plot(arch_means[:,0], arch_means[:,1], '-', label=classifier)
                plt.fill_between(arch_means[:,0], arch_means[:,1] - stds, arch_means[:,1] + stds, alpha=0.2)
                
                if classifier in CLASSIFIERS_W_PARAMS:
                    # Write what classifier parameter results were plotted
                    for txt,(x,y) in zip(classifier_params[classifier].values(),arch_means):
                        plt.annotate(txt, (x, y))
                
            plt.legend()
            plt.xticks(arch_means[:,0])
            plt.title("Final classification accuracy for different number of archetypes")
            plt.xlabel("Number of archetypes")
            plt.ylabel("Accuracy")
            
            output_plot_path = os.path.join(raw_result_path, "plots")
            if not os.path.exists(output_plot_path):
                os.mkdir(output_plot_path)
            plt.savefig(os.path.join(output_plot_path,"class_accs_per_archetype.png"))   
            #plt.show() 
        else:
            raise NotImplementedError("Seedwise classification accuracy has not been implemented yet")
        
        
if __name__ == "__main__":
    Performance_evaluator.plot_results(r"Classifier\cleaned_up\test_results\multiconditional\6_08-27-2023",True)
    
    