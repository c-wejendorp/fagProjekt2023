import json
def createArguments_evening_run():    
    modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"]]
    for idx,modalityComb in enumerate(modalityCombs):    
        # i will implement that we read this from json file later 
        arguments = {    
        'subjects': list(range(1,17)),
        #'iterations': 5,
        'iterations': 300,
        'archeTypeIntevalStart': 22,
        'archeTypeIntevalStop': 40,
        'archeTypeStepSize': 2,
        #'seeds': list(range(0, 11, 10)),
        'seeds': list(range(0, 91, 10)),
        'lossRobust': True,
        'modalities': modalityComb,
                                    }
        # save to json file
        with open(f'MMAA/HPC/arguments_evening_run/arguments{idx}.json', 'w') as fp:
            json.dump(arguments, fp)

if __name__ == "__main__":
    createArguments_evening_run() 
