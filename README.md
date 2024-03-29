# Resume Scorer
It would be interesting to see how much a resume matches a job role just by comparing the job description with the contents of the resume. A state-of-the-art NLP model is used to create word embeddings of a resume file and the embeddings is compared to an embedding vector of a job description. Faiss index, a distance metric, is used to compute the distance between the two embedding vectors. The smaller the distance, the closer the resume matches the job description.

⭐️ Star this repo if you like it ⭐️

## Applications
For a recruiter, this could be used to automatically scan through the hundreds, or possibly thousands, of CV files for a job opening. The resumes which closely reflect the job role based on the job description would be the winning candidates. It would reduce the brute force approach of looking through individual resumes just to know which one would be a good fit for the role. This automated first selection process would make hiring process faster and provide more time to prepare interviews for the candidates. 

A job seeker would be able to use this tool to compare how much their resume fits a role. Seeing how lacking a resume is compared to a job description would give room to make the necessary adjustments to make the CV more tailored to the role than using a one-size-fits-all CV for all job applications. With a resume scorer, you would get an immediate feedback on your resume even before sending out your application. The feedback (score) you get would be similar to the outcome of your job application since most companies use ATS to screen through applicants CVs, which basically work based on comparing text embedding vectors.

## About the data
The data used in this repo was obtained from [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). 

## Installation
Clone this repository and change your working directory to this repo
```
git clone https://github.com/DozieSixtus/Resume-Scorer.git
cd Resume-Scorer
```
Install the dependencies required
```
pip install -r requirements.txt
```

## Inferencing
To implement resume scorer on your device, `run.py` contains the code for inferencing. `run.py` takes three parameters:

&emsp; `--cvPath`: this parameter takes the path to the FOLDER where the cv file is located. The cv file needs to be a pdf file format. The path object type should be a `str` object.

&emsp; `--jobDesc`: this parameter is the ABSOLUTE PATH to a text file (.txt) containing the job description for the role that needs to be scored. The path object type should be a `str` object.

&emsp; `--topCV`: this `integer` parameter is used to determine the top n resumes that matches a job description. `--topCV` would take a value of `1` if an individual wants to obtain a score from a single CV, but the value could be greater than 1 if multiple CVs are being scored. So a recruiter can set the value of `--topCV` to be the number of best-matching CVs desired.

```
python .\scripts\run.py --cvPath "resumeFolderPath" --jobDesc "jobDescriptionFilePath" --topCV 1
```
A csv file saved as "jobScores.csv" containing the scores would be created in the models folder after running the code above.
### Licence
See [LICENSE](LICENSE) for details.