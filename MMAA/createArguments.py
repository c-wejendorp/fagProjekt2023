import json

if __name__ == "__main__":   

    modalityCombs = [["eeg", "meg", "fmri"],["eeg", "meg"], ["eeg", "fmri"], ["meg", "fmri"]]
    for idx,modalityComb in enumerate(modalityCombs):    
        # i will implement that we read this from json file later 
        arguments = {    
        'subjects': list(range(1,17)),
        'iterations': 5,
        #'iterations': 300,
        'archeTypeIntevalStart': 2,
        'archeTypeIntevalStop': 20,
        'archeTypeStepSize': 2,
        'seeds': list(range(0, 11, 10)),
        #'seeds': list(range(0, 91, 10)),
        'lossRobust': True,
        'modalities': modalityComb,
                                    }
        # save to json file
        with open(f'MMAA/arguments{idx}.json', 'w') as fp:
            json.dump(arguments, fp)


