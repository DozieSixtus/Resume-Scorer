import os, sys, argparse
from pathlib import Path
from parse_text import readPdf
from prep_csv import remove_encodings, readFile, partitionText
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from job_desc import loadJobDesc

sys.path.append(os.getcwd() + '\\models')
from semantic_search import queryTexts, get_embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--cvPath', type=str, required=True)
parser.add_argument('--jobDesc', type=str, required=True)
parser.add_argument('--topCV', type=int, required=True)
args = parser.parse_args()

pdfPath = args.cvPath
jobDescPath = args.jobDesc
topK = args.topCV

df = readPdf(pdfPath)
savePath = Path(os.getcwd()+'\models')
os.makedirs(savePath, exist_ok=True)
df.to_csv(os.path.join(savePath, 'resume.csv'))

filePath = os.getcwd()+'\\models\\resume.csv'
df = remove_encodings(readFile(filePath =filePath))

dataset = dict()
dataset['cvContent'] = partitionText()
dataset = Dataset.from_dict(dataset)

embedding = get_embeddings(dataset['cvContent'][0])

embeddings_dataset = dataset.map(
    lambda x: {"embeddings": get_embeddings(x["cvContent"]).numpy()[0]}
)

embeddings_dataset.add_faiss_index(column="embeddings")

if __name__ == '__main__':
    print('Running ...')
    queryTexts(embDataset=embeddings_dataset,jobDesc=loadJobDesc('txt', jobDescPath), topK=topK)