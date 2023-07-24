import pandas as pd
import os, shutil, numpy as np, re, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from collections import Counter


pd.set_option("display.max_colwidth", 200)


def readFile(filePath=None):
    return pd.read_csv(filePath)

def remove_encodings(csv):
    csv['Content'] = csv['Content'].apply(lambda x: ''.join(a for c in x for a in c))
    csv['Content'] = csv['Content'].apply(lambda x: x[2:-2])
    csv['Content'] = csv['Content'].apply(lambda x: x.replace('\\n', ' [sub title] '))
    return csv

def partitionText():
    df = remove_encodings(readFile(filePath=os.getcwd()+'\\models\\resume.csv'))
    texts = df['Content'].values.tolist()
    subStrings = []
    for text in texts:
        #print(re.findall('\[sub title\]\s?(.{20})\s?\[sub title\]', text))
        subStrings.append(re.findall('\[sub title\]\s?(.{6,20})\s?\[sub title\]', text))
    subStrings = [x for strs in subStrings for x in strs if ' [sub title] ' not in x]
    subStrings = Counter(subStrings).most_common()
    commonWords = [word.lower() for word,_ in subStrings]
    commons = ['january','february','march','april','may','june','july',
              'august','september','october','november','december',
              'jan','feb','mar','apr','jun','jul','aug','sep','oct','nov','dec',
              'city', 'state']

    keyPhrases = []
    with open('scripts\\key phrases.txt') as file:
        for line in file:
            keyPhrases.append(line.strip())
    #print(keyPhrases)

    for i,keyPhrase in enumerate(keyPhrases):
        if i == 0:
            pattern = keyPhrase
        pattern = pattern+'|'+keyPhrase
    
    nltk.download('stopwords')
    stopWords = list(set(stopwords.words('english')))
    nltk.download('wordnet')
    lemma = WordNetLemmatizer()

    textsSplits = [re.split(f'\[sub title\]\s?{pattern}\s?\[sub title\]', text) for text in texts]
    textsSplits = [[x.replace('[sub title] ', '') for x in y][1:] for y in textsSplits]
    textsSplits = [''.join([i if ord(str(i)) < 128 else '' for x in y for i in x]) for y in textsSplits]
    textsSplits = [x.split() for x in textsSplits]
    textsSplits = [''.join([x.lower()+' ' for x in y if x.lower() not in commonWords[:100]+commons]) 
                   for y in textsSplits]
    textsSplits = [''.join([x for x in y if not x.isdigit()]) for y in textsSplits]
    textsSplits = [''.join(lemma.lemmatize(x)+' ' for x in y.split()) for y in textsSplits]
    
    #for i in range(3):
        #print(f"for split {i}:", '*'*55, '\n',textsSplits[i])
    
    return textsSplits


"""if __name__ == '__main__':
    filePath = os.getcwd()+'\\models\\resume.csv'
    df = remove_encodings(readFile(filePath =filePath))
    partitionText()"""