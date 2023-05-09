import pandas as pd
import os, shutil, numpy as np, re
from pathlib import Path
from collections import Counter


pd.set_option("display.max_colwidth", 200)

def readFile(filePath=None):
    return pd.read_csv(filePath)

def remove_encodings(csv):
    csv['Content'] = csv['Content'].apply(lambda x: ''.join(a for c in x for a in c))
    csv['Content'] = csv['Content'].apply(lambda x: x[2:-2])
    csv['Content'] = csv['Content'].apply(lambda x: x.replace('\\n', ' [sub title] '))
    #print(csv.head(1).to_string())
    return csv

def partitionText(df):
    texts = df['Content'].values.tolist()
    subStrings = []
    for text in texts:
        #print(re.findall('\[sub title\]\s?(.{20})\s?\[sub title\]', text))
        subStrings.append(re.findall('\[sub title\]\s?(.{6,20})\s?\[sub title\]', text))
    subStrings = [x for strs in subStrings for x in strs if ' [sub title] ' not in x]
    subStrings = Counter(subStrings).most_common()
    print(subStrings[:200])

    keyPhrases = []
    with open('scripts\\key phrases.txt') as file:
        for line in file:
            keyPhrases.append(line.strip())
    #print(keyPhrases)

    

if __name__ == '__main__':
    filePath = os.getcwd()+'\\models\\resume.csv'
    df = remove_encodings(readFile(filePath =filePath))
    partitionText(df)