from datasets import load_dataset, Dataset
import os, pandas as pd, torch
import sys
from transformers import AutoTokenizer, TFAutoModel
sys.path.append(os.getcwd() + '\\scripts')
import prep_csv

dataset = dict()
dataset['cvContent'] = prep_csv.partitionText()[:200]
dataset = Dataset.from_dict(dataset)
print(dataset)

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt = True)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embedding = get_embeddings(dataset['cvContent'][0])
print(embedding.shape)

embeddings_dataset = dataset.map(
    lambda x: {"embeddings": get_embeddings(x["cvContent"]).numpy()[0]}, batched=True
)
print(embeddings_dataset)

embeddings_dataset.add_faiss_index(column="embeddings")