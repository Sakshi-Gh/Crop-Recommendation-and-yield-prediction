# -*- coding: utf-8 -*-
"""crop_recommendation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ievH0YGOKA1pBy1QMdtAEhKUzoN7FQnX
"""

# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("train.csv")

df.head()

df.shape

df.describe()

df.columns

df.isna().sum()

df.dtypes

#unique crops
df['Crop'].unique()

df['Crop'].value_counts()

features = df[['N', 'P','K','pH', 'rainfall','temperature']]
target = df['Crop']
labels = df['Crop']

# Splitting into train and test data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

accuracy = []
model_names = []

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
accuracy.append(x)
model_names.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# Cross validation score (Logistic Regression)
from sklearn.model_selection import cross_val_score
score = cross_val_score(LogReg,features,target,cv=5)
score

#Not saving Logistic Regression -- less accuracy

from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
accuracy.append(x)
model_names.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))

from sklearn.model_selection import cross_val_score
score = cross_val_score(DecisionTree, features, target,cv=5)
score

import pickle
DT_pkl_filename = 'DecisionTree.pkl'
# Open the file to save as pkl file
DT_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(DecisionTree, DT_Model_pkl)
# Close the pickle instances
DT_Model_pkl.close()

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
accuracy.append(x)
model_names.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score

import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()

from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
accuracy.append(x)
model_names.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,features,target,cv=5)
score

import pickle
# Dump the trained Naive Bayes classifier with Pickle
NB_pkl_filename = 'NBClassifier.pkl'
# Open the file to save as pkl file
NB_Model_pkl = open(NB_pkl_filename, 'wb')
pickle.dump(NaiveBayes, NB_Model_pkl)
# Close the pickle instances
NB_Model_pkl.close()

from sklearn.svm import SVC

SVM = SVC(gamma='auto')

SVM.fit(Xtrain,Ytrain)

predicted_values = SVM.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
accuracy.append(x)
model_names.append('SVM')
print("SVM's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# Cross validation score (SVM)
score = cross_val_score(SVM,features,target,cv=5)
score

import pickle
# Dump the trained SVM classifier with Pickle
SV_pkl_filename = 'SVM.pkl'
# Open the file to save as pkl file
SV_Model_pkl = open(SV_pkl_filename, 'wb')
pickle.dump(SVM, SV_Model_pkl)
# Close the pickle instances
SV_Model_pkl.close()

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Encode the target variable
label_encoder = LabelEncoder()
Ytrain_encoded = label_encoder.fit_transform(Ytrain)
Ytest_encoded = label_encoder.transform(Ytest)

# Initialize the XGBoost classifier
XGB = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train the model
XGB.fit(Xtrain, Ytrain_encoded)

# Predict on the test data
predicted_values_encoded = XGB.predict(Xtest)

# Decode the predictions back to original labels
predicted_values = label_encoder.inverse_transform(predicted_values_encoded)

# Calculate accuracy
x = accuracy_score(Ytest, predicted_values)
accuracy.append(x)
model_names.append('XGBoost')
print("XGBoost's Accuracy is: ", x)

# Print classification report
print(classification_report(Ytest, predicted_values))

# Dump the trained XGB classifier with Pickle
XG_pkl_filename = 'XGB.pkl'
# Open the file to save as pkl file
XG_Model_pkl = open(XG_pkl_filename, 'wb')
pickle.dump(XGB, XG_Model_pkl)
# Close the pickle instances
XG_Model_pkl.close()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Initialize the Decision Tree model
dt_tuned = DecisionTreeClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],  # Criteria for split quality
    'max_depth': [None, 10, 20, 30],  # Depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split
    'min_samples_leaf': [1, 2, 5],    # Minimum samples required at a leaf
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=dt_tuned,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=1
)

# Train the model with GridSearchCV
grid_search.fit(Xtrain, Ytrain)

# Best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Predict on the test data using the best model
predicted_values = best_model.predict(Xtest)

# Calculate accuracy
x = accuracy_score(Ytest, predicted_values)
accuracy.append(x)
model_names.append('Decision Tree (Tuned)')
print("Tuned Decision Tree's Accuracy is: ", x)

# Print classification report
print(classification_report(Ytest, predicted_values))

# Dump the trained Hypertuned Decision Tree classifier with Pickle
DT_tuned_pkl_filename = 'DT_tuned.pkl'
# Open the file to save as pkl file
DT_tuned_Model_pkl = open(DT_tuned_pkl_filename, 'wb')
pickle.dump(dt_tuned, DT_tuned_Model_pkl)
# Close the pickle instances
DT_tuned_Model_pkl.close()

plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = accuracy,y = model_names,palette='dark')

accuracy_models = dict(zip(model_names, accuracy))
for model, accuracy in accuracy_models.items():
    print (f'{model} has accuracy {accuracy}.')