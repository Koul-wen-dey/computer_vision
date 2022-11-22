import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
# Path = r'./Dataset_OpenCvDl_Hw2_Q5/training_dataset/Cat/'
# Path2 = r'./Dataset_OpenCvDl_Hw2_Q5/training_dataset/Dog/'
Path = r'./Dataset_OpenCvDl_Hw2_Q5/validation_dataset/Cat/'
Path2 = r'./Dataset_OpenCvDl_Hw2_Q5/validation_dataset/Dog/'

file = pd.DataFrame(columns=['path','label'])
count = 0
count2 = 0
for path in os.listdir(Path):
    if os.path.isfile(os.path.join(Path, path)):
        file.loc[len(file)] = [os.path.join(Path, path),1]
        count += 1
for path in os.listdir(Path2):
    if os.path.isfile(os.path.join(Path2, path)):
        file.loc[len(file)] = [os.path.join(Path2, path),0]
        count2 += 1
x = ['Cat','Dog']
y = [count,count2]
print(count,count2)
file = shuffle(file)
file.to_csv('validset.csv',index=False)
# file.to_csv('dataset.csv',index=False)
# plt.title('Class Distribution')
# plt.bar(x,y)
# plt.savefig('fig2.jpg')