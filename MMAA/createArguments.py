import json

if __name__ == "__main__":   
        
    # i will implement that we read this from json file later 
    arguments = {    
    'subjects': list(range(1,17)),
    'iterations': 200,
    'archeTypeIntevalStart': 2,
    'archeTypeIntevalStop': 20,
    'archeTypeStepSize': 2,
    'seeds': list(range(0, 101, 10)),
    'lossRobust': True,
    'split': 0                             }
    # save to json file
    with open('MMAA/arguments.json', 'w') as fp:
        json.dump(arguments, fp)


