import sys
import getopt
from loadData import Real_Data
from MMA_model_NEW import MMAA, trainModel
import ast

def parseArgs(argv):
    # Check if at least one argument is provided
    if len(sys.argv) != 7:
        print("Please provide the following arguments in this specfic order <subjectIdxList> <numArcheTypes> <archeTypeStepSize> <seed> <lossFunction> <split>")
        sys.exit(1)

    # Access the command-line arguments
    arguments = {
    'subjects': ast.literal_eval(sys.argv[1]),
    'numArcheTypes': sys.argv[2],
    'archeTypeStepSize': sys.argv[3],
    'seed': sys.argv[4],
    'lossFunction': sys.argv[5],
    'split': sys.argv[6]
                         }

    return arguments

def trainMultipleModels(numArcheTypes, numSeeds, lossFunction, split):
    pass 
 
if __name__ == "__main__":
    #if we give inputs from the command line
    if len(sys.argv) > 1:
        arguments = parseArgs(sys.argv)
    else:
    #if no input given, use these default values
        arguments = {
        'subjects': range(1, 17),
        'numArcheTypes': 16,
        'archeTypeStepSize': 2,
        'seed': 0,
        'lossFunction': "MLE",
        'split': 0                             }
        
    X = Real_Data(subjects=arguments["subjects"],split=arguments["split"])
    



