# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:33:05 2020

@author: Anandhu Sanu
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #importing decision tree classifier
from sklearn.model_selection import train_test_split #importing train_test_split function
from sklearn.metrics import accuracy_score#importing metrics for accuracy calculation (confusion matrix)
from sklearn.ensemble import BaggingClassifier#bagging combines the results of multipls models to get a generalized result. 
from sklearn.metrics import classification_report, confusion_matrix
#reading the dataset
fraud=pd.read_csv("F:\\ExcelR\\excelRASS\\ass4 decision trees\\Fraud_Check.csv")
fraud.head()
#viewing the types
fraud.dtypes
#-------------------converting categorical data-----------------------------------# 
fraud['Risky_1'] = fraud.Taxable_Income.map(lambda x: 1 if x <= 30000 else 0)
fraud['Undergrad']=fraud['Undergrad'].astype('category')
fraud['Marital_Status']=fraud['Marital_Status'].astype('category')
fraud['Urban']=fraud['Urban'].astype('category')
fraud.dtypes
fraud.head()
#label encoding to convert categorical values into numeric.
fraud['Undergrad']=fraud['Undergrad'].cat.codes
fraud['Urban']=fraud['Urban'].cat.codes
fraud['Marital_Status']=fraud['Marital_Status'].cat.codes
fraud.head()
#------------------------ setting feature and target variables -------------------------------------------------------------#
feature_cols=['Undergrad','Marital_Status','City_Population','Work_Experience','Urban']
#x = fraud.drop(['Taxable_Income','Risky_1'], axis=1)
x = fraud[feature_cols]
y = fraud.Risky_1
print(x)
print(y)
#------------------------ splitting into train and test data -------------------------------------------------------------#
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.20,random_state=1)
#-------------------------building decision tree model-----------------------------# 
fraudmodel =  BaggingClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object

fraudmodel = fraudmodel.fit(x_train,y_train) #train decision tree
y_predict = fraudmodel.predict(x_test)
#-----------Finding the accuracy------------------------------------#
print("Accuracy : ", accuracy_score(y_test,y_predict)*100 )
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))