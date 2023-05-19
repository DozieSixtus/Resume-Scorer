from datasets import load_dataset, Dataset
import os, pandas as pd, torch, tensorflow as tf, copy
import sys
from transformers import AutoTokenizer, TFAutoModel
sys.path.append(os.getcwd() + '\\scripts')
import prep_csv

dataset = dict()
dataset['cvContent'] = prep_csv.partitionText()[:2000]
dataset = Dataset.from_dict(dataset)
print(dataset)

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt = True)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def chunk_examples(examples):
    chunks = []
    for sentence in examples["cvContent"]:
        chunks += [sentence[i:i + 50] for i in range(0, len(sentence), 50)]
    return {"cvContent": chunks}

dataset = dataset.map(chunk_examples, batched=True)

def get_embeddings(text_list):
    batch_size = 8
    encoded_inputs = dict()
    for i in range(0, len(text_list), batch_size):
        #print("the text is: ", text_list[i:i+batch_size])
        encoded_input = tokenizer(
            text_list[i:i+batch_size], padding=True, truncation=True, return_tensors="tf"
        )
        print('Encoded input:','*'*55,'\n', encoded_input) # line
        temp_encoded_inputs = {k: v for k, v in encoded_input.items()}
        if i == 0:
            encoded_inputs = copy.deepcopy(temp_encoded_inputs)
        else:
            for k, v in temp_encoded_inputs.items():
                print("v: ", v.get_shape().as_list())
                print("k: ", encoded_inputs[k].get_shape().as_list())
                if v.get_shape().as_list()[1] < encoded_inputs[k].get_shape().as_list()[1]:
                    v = tf.concat([v,tf.zeros([v.get_shape().as_list()[0],
                                              encoded_inputs[k].get_shape().as_list()[1]-v.get_shape().as_list()[1]],
                                              tf.int32)], 
                                              1)
                encoded_inputs[k] = tf.concat([encoded_inputs[k],v],0)
        print('Encoded inputs:','*'*55, '\n', encoded_inputs.items())  # line
    model_output = model(**encoded_inputs)
    print('*'*55, '\n', type(model_output)) # line
    return cls_pooling(model_output)

embedding = get_embeddings(dataset['cvContent'])
print(embedding.shape)