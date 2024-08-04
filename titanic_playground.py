#%%
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
#%%
file_EXT = 'train.csv'
foldName = Path('C:/Users/nicho/Documents/KAGGLE')  # Using forward slashes or pathlib Path
fileName = foldName / file_EXT

#%%
print(fileName)
#%%
# Load the CSV file into a DataFrame
df = pd.read_csv(fileName)

# Display the first few rows of the DataFrame
print(df.head())

#%% PRE-PROCESS DATA
label_encoder = LabelEncoder()
# Fit and transform the 'city' column
df['Sex'] = label_encoder.fit_transform(df['Sex'])
# print(df)
data_train = (df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']])
df_cleaned = data_train.dropna()
#%%
target_vector = pd.to_numeric(df_cleaned['Survived'])
data_train = (df_cleaned[['Pclass','Sex','Age','SibSp','Parch','Fare']])
data_train.apply(pd.to_numeric, errors='ignore')
# print(data_train)
# %%
model = LogisticRegression(random_state=0).fit(data_train,target_vector)

