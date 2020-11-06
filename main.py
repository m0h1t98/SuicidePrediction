# laod the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# load the dataset

df = pd.read_csv('foreveralone.csv')

cols = ['gender', 'sexuallity', 'friends', 'age', 'income', 'bodyweight', 'virgin', 'social_fear', 'employment', 'depressed', 'attempt_suicide']

df1 = df[cols]
cat_cols = ['gender', 'sexuallity', 'bodyweight', 'virgin', 'social_fear', 'depressed', 'employment', 'attempt_suicide']
df1['gender'].replace('Transgender male', 'Male', inplace=True)
df1['gender'].replace('Transgender female', 'Female', inplace=True)

for column in df1.columns:
    if df1[column].dtype == 'object':
        df1[column] = LabelEncoder().fit_transform(df1[column])

final_df = df1

X = final_df.drop('attempt_suicide', axis=1)
y = final_df['attempt_suicide']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=92)

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('Confustion Matrix : \n\n', confusion_matrix(y_test,  rf.predict(X_test)))
print('\n Accuracy Score : ',   accuracy_score(y_test,  rf.predict(X_test)))
print('\n Classification Report : \n \n',classification_report(y_test, rf.predict(X_test)))


# saving the model to the local file system
filename = 'model.pkl'
joblib.dump(rf, open(filename, 'wb'))

joblib.load(open('model.pkl','rb'))