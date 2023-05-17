from datasets import load_dataset, Dataset
import os, pandas as pd
import sys
from transformers import AutoTokenizer, TFAutoModel
#***************** correct path name
sys.path.append('C:/Users/Dozie Sixtus/Documents/resume scorer/scripts')
import prep_csv

dataset = dict()
dataset['cvContent'] = prep_csv.partitionText()
dataset = Dataset.from_dict(dataset)
print(dataset)

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt = True)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]