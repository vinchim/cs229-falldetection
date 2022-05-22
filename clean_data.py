
import numpy as np
import pandas as pd
import csv

def main():
    data = pd.read_csv("data.csv")
    print(data)

    with open('data_clean.csv', 'wb') as out:
        writer = csv.writer(out)
        # go through each row of the data
        for row in data.iterrows():
            # extract the file name
            # example: dataset/rgb_0001.png
            file = row[1]["file_name"]
            names = file.split("/", 1)
            group = int(names[0])
            index = int(names[1].split("_", 1)[1].split(".",1)[0])
            if index <= group:
                writer.writerow(row)

if __name__ == "__main__":
    main()