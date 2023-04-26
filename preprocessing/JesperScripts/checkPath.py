import os
import sys
def checkPath():
    paths = [r"C:\Users\chwe\Desktop\projectModule", #Christoffer
            r"", #Danina
            r""] #Helena
    #find the first path that exists
    for path in paths:
        if os.path.exists(path):
            sys.path.insert(0,path)
            break
        raise Exception("No valid path found")
    
    sys.path.insert(0,r"C:\Users\chwe\Desktop\projectModule")
 