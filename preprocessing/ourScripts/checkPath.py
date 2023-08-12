import os
import sys
def checkPath():
    paths = [r"C:\Users\chwe\Desktop\projectModule", #Christoffer
            r"C:\University\fagProjekt2023\projectModule", #Danina
            "/Volumes/HelenaKeitumEkstern/projectModule"] #Helena
    #find the first path that exists
    exists = False
    for path in paths:
        if os.path.exists(path):
            exists = True
            sys.path.insert(0,path)
            break
    if not exists:
        raise Exception("No valid path found")