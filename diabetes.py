# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# %matplotlib inline

data = pd.read_csv('diabetes_data.csv')
#data.tail()

data=data.rename(columns = {'class': 'diabetes'}, inplace = False)

#data.isnull().values.any()

#data.info()

data['diabetes'] = data['diabetes'].replace({'Positive': 1,'Negative':0})

data['Gender'] = data['Gender'].replace({'Male': 1,'Female':0})

data[['Polyuria','Polydipsia']] = data[['Polyuria','Polydipsia']].replace({'Yes': 1,'No':0})

data[['sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']] = data[['sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']].replace({'Yes': 1,'No':0})

#data.head()

#data.corr().T

data.drop(['Alopecia'], axis='columns', inplace=True)

import seaborn as sns
#get correlations of each features in dataset
#corrmat = data.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
#plot heat map
#g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#data.head()

X = data.iloc[:, :-1]

#X.head()

y = data.iloc[:,-1:]

#y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)

#X_train.head()

#X_test.shape

#y_train.shape

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
#print(accuracy_score(y_test,y_pred))

#print(classification_report(y_test,y_pred))

from sklearn.model_selection import cross_val_score
desiciontree = DecisionTreeClassifier()
#print(cross_val_score(desiciontree, X_test, y_test, cv=10, scoring = 'accuracy').mean())


import pickle
import os

if not os.path.exists('model'):
    os.makedirs('model')
    
MODEL_PATH = "model/classify.sav"
pickle.dump(clf, open(MODEL_PATH, 'wb'))

model=pickle.load(open('model.pkl','rb'))


