import re

def loadJobDesc():
    desc = input("Enter the job description: ")
    desc = ''.join(x if ord(str(x))<128 else '' for x in desc)
    desc = desc.splitlines()
    desc = [re.split('\. ',x) for x in desc]
    print(desc)

if __name__ == '__main__':
    loadJobDesc()