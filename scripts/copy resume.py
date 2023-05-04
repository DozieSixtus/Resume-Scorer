import os, shutil
from pathlib import Path

def getAllResume(fileLoc = None):
    for (root, dirs, resumes) in os.walk(fileLoc):
        for resume in resumes:
            newPath = os.getcwd()+'\models\data'
            os.makedirs(newPath, exist_ok=True)
            shutil.copy(Path(os.path.join(root, resume)), 
                        os.path.join(newPath,resume))


if __name__ == '__main__':
    
    getAllResume(Path(os.getcwd()+'\data'))