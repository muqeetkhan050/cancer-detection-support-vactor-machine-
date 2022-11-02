# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:46:53 2022

@author: Muqeet
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('cell_samples.csv')

data.head()

data.shape

data.size

data.count()

data['Class'].value_counts()

malignent=data[data['Class']==4][0:200]
benign=data[data['Class']==2][0:200]

axes=benign.plot(kind='scatter',x='Clump',y='Unifsize',color='blue',label='benign')

sns.scatterplot(data=data,x='Clump',y='UnifSize',hue="region")

data.dtypes

data=data[pd.to_numeric(data['BareNuc'],errors='coerce').notnull()]
data['BareNuc']=data['BareNuc'].astype('int')

data.dtypes

data.columns

feature=data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]


x=np.asarray(feature )

y=np.array(data['Class'])

x[0:5]
y[0:5]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



from sklearn import svm
classifier=svm.SVC(kernel='linear',gamma='auto',C=2)
classifier.fit(x_train,y_train)

y_predict=classifier.predict(x_test)


from sklearn.metrics import classification_report

classification_report(y_test, y_predict)

