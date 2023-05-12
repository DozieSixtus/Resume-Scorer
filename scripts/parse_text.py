import PyPDF2, os
from pathlib import Path
import pandas as pd

def readPdf(folderPath = None):
    pdfDict = {}
    for file in os.listdir(folderPath):
        reader = PyPDF2.PdfReader(os.path.join(folderPath, file))
        fileContent = []
        print(file)
        for i in range(len(reader.pages)):
            fileContent.append(reader.pages[i].extract_text())
        pdfDict[file] = fileContent
        
    dataset = pd.DataFrame()
    dataset['Content'] = pdfDict.values()
    
    return dataset

if __name__ == '__main__':
    df = readPdf(Path(os.getcwd() +'\models\data'))
    savePath = Path(os.getcwd()+'\models')
    os.makedirs(savePath, exist_ok=True)
    df.to_csv(os.path.join(savePath, 'resume.csv'))