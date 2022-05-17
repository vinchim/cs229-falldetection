
import numpy as np
import pandas as pd
import csv

def main():
    data = pd.read_csv("data.csv")
    print(data)
    num_files, features = data.shape 
    labelsDict = {}
    datasets = [1301, 1790, 722, 1378, 1392, 807, 758, 1843, 569, 1260, 489, 731, 1219, 1954, 581, 1176, 2123, 832, 786, 925]
    path = "/data/xyz/labels.csv"
    answer = []
    #make all the dataset labels into dataframes and put them in a dictionary
    for dset in datasets:
        currpath = path.replace("xyz", str(dset))
        currData = pd.read_csv(currpath)
        currData = currData.astype({"index":"int","class":"int"})
        labelsDict[dset] = currData
        
    #go through each row of the data and add their corresponding label to the array

    # with open(data, "r") as f:
    #     reader = csv.reader(f, delimiter="\t")
    #     for i, row in enumerate(reader):
    #         currFile = row[0]
    #         names = currFile.split("/", 1)
    #         currDsetName = names[0]
    #         currDset = labelsDict[int(currDsetName)]

    #         temp = names[1].split("_", 1)
    #         temp2 = temp[1].split(".",1)
    #         index = int(temp2[0])
    #         currLabel = currDset.loc[currDset['index'] == index,'class'].values[0]
    #         answer.append(currLabel)
           
    for row in data.iterrows():
        #extract the file name
        #example: dataset/rgb_0001.png
        currFile = row[0]
        print(currFile)
        names = currFile.split("/", 1) 
        currDsetName = names[0]
        currDset = labelsDict[int(currDsetName)]
        
        temp = names[1].split("_", 1)
        temp2 = temp[1].split(".",1)
        index = int(temp2[0])
        currLabel = currDset.loc[currDset['index'] == index,'class'].values[0]
        answer.append(currLabel)

    a = np.array(answer)
    print(a.T)
    np.savetxt('processedLabels.csv', a.T, delimiter=",")

if __name__ == "__main__":
    main()