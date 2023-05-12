from datasets import load_dataset, Dataset
import os
import sys
sys.path.append('C:/Users/Dozie Sixtus/Documents/resume scorer/scripts')
import prep_csv

dataset = prep_csv.partitionText()
dataset = [x.append(x) for y in dataset for x in y]
#dataset = Dataset.from_list(dataset)
print(dataset)