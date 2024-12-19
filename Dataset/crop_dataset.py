import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
crop=pd.read_csv("C:/Users/Md Ganim/Desktop/Program/AI_project/Final/Dataset/Crop_recommendation.csv")
crop.shape
crop.head()
crop['N'].describe()
crop['P'].describe()
crop['K'].describe()
crop['label'].describe()
crop['label'].unique()
soil=pd.read_csv("C:/Users/Md Ganim/Desktop/Program/AI_project/Final/Dataset/soil.csv")
soil.head()
soil.rename(columns = {'crop':'label'}, inplace = True) 
soil.head()
final = pd.merge(crop, soil, on ='label') 
final.head()
final.describe()
final = final[['N', 'P', 'K','temperature', 'humidity', 'ph', 'rainfall', 'soil','label']]
final.to_csv("C:/Users/Md Ganim/Desktop/Program/AI_project/Final/Dataset/FinalCrop_Reommenationn.csv",index=False)