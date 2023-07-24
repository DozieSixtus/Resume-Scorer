import re, nltk
from nltk.corpus import stopwords
import prep_csv
from nltk.stem import WordNetLemmatizer

def loadJobDesc(textType=None, path=None):
    if textType == 'input':
        desc = input("Enter the job description: ")
    elif textType == None:
        return ''
    elif textType == 'txt':
        desc = []
        with open(path, 'r') as f:
            for line in f:
                desc.append(line.strip())
            desc = ''.join(x+' ' for x in desc)
    commons = ['january','february','march','april','may','june','july',
            'august','september','october','november','december',
            'jan','feb','mar','apr','jun','jul','aug','sep','oct','nov','dec',
            'city', 'state']
    
    stopWords = list(set(stopwords.words('english')))
    lemma = WordNetLemmatizer()

    desc = ''.join(x if ord(str(x))<128 else '' for x in desc)
    desc = ''.join(x.lower()+' ' for x in desc.split() if x.lower()
                    not in stopWords+commons and not x.isdigit())
    desc = ''.join(lemma.lemmatize(x)+' ' for x in desc.split())
    return desc


"""if __name__ == '__main__':
    loadJobDesc('txt',"C:\\Users\Dozie Sixtus\\Documents\\resume scorer\\models\\data\\job.txt")"""