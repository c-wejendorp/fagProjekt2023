# Classification

Classification scripts used to quantitavely evaluate results of the archetypal analysis.

## Structure
- [`main_classifier.py`](./main_classifier.py): main python file to run to get the classification results. Current choices of classifiers are PCA w LR / no PCA w LR-w-regularization / baseline that will randomly choose between the most appearing training labels.
- [`config.yaml`](./config.yaml): file that holds all the configurations that `main_classifier.py` uses. check the file for more info.
- [`classification_evaluator.py`](classification_evaluator.py): contains a class with static methods that can used for evaluating classification results.
- [`pca.py`](./pca.py): contains a function that performs pca over given data 

## Submitting jobs to HPC
There are two available methods:
- Sending multiple batch jobs 
- Requesting multiple cores to compute [`main_classifier.py`](./main_classifier.py) as parallelization has already been set up

__IMPORTANT!__
In order to choose between the two do the following:
1. change [`config.yaml`](./config.yaml) accordingly under the HPC section.
2. run INSERT SCRIPT to auto generate bash script(s)
3. Make sure to submit the correct bash script(s) to HPC
    - The command to submit a script is `INSERT COMMAND`

## Further evaluation of classifier results on specific archetypes
McNemar's test and confusion matrices can be done by changing [`config.yaml`](./config.yaml) accordingly under the Further Evaluations section.

Note that the p-values are Bonferroni corrected as a default, and can only be changed by changing INSERT SCRIPT 
Just submit INSERT SCRIPT to HPC by running `INSERT COMMAND` and voila! 

## Extra stuff and to do's
- there still needs bash scripts set up to automate the process, but the main file runs perfectly fine with its configs already
- make further evaluations python script using existing functions
