import PyPDF2, os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def readPdf(folderPath = None):
    pdfDict = {}
    for file in tqdm(os.listdir(folderPath)):
        try:
            if file[-3:] == 'pdf' or 'PDF':
                reader = PyPDF2.PdfReader(os.path.join(folderPath, file))
                fileContent = []
                #print(file)
                for i in range(len(reader.pages)):
                    fileContent.append(reader.pages[i].extract_text())
                pdfDict[file] = fileContent
        except:
            pass
        
    dataset = pd.DataFrame()
    dataset['Content'] = pdfDict.values()
    
    return dataset

"""if __name__ == '__main__':
    df = readPdf(Path(os.getcwd() +'\models\data'))
    savePath = Path(os.getcwd()+'\models')
    os.makedirs(savePath, exist_ok=True)
    df.to_csv(os.path.join(savePath, 'resume.csv'))"""