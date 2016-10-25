import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg') 
plt.style.use('ggplot')

#names = ['Patient_ID','Online_Follower','LinkedIn_Shared','Twitter_Shared','Facebook_Shared','Income','Education_Score','Age','First_Interaction','City_Type','Employer_Category']
df = pd.read_csv("/home/raghunandangupta/Downloads/Train_2/Train/Patient_Profile.csv",",")

df[df.Income == 'None'] = 0
df[df.Education_Score == 'None'] = 0
df[df.Age == 'None'] = 0

print df['Employer_Category']
df['City_Type'].replace(np.NaN, 'Z', inplace=True)
df['City_Type'].replace(0, 'Z', inplace=True)
df['Employer_Category'].replace(0, 'Others', inplace=True)

#df[['Income','Education_Score','Age','City_Type','Employer_Category']] = df[['Income','Education_Score','Age','City_Type','Employer_Category']].apply(lambda x: pd.to_numeric(x, errors='coerce'))


print df.head(15)

