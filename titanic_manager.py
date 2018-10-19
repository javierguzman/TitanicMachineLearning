# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from titanic_imputer import TitanicImputer

#Import data
train_data = pd.read_csv('train.csv')
y_train = train_data['Survived'].values.reshape(len(train_data), 1)
X_train = train_data[['Pclass', 'Sex', 'Age','SibSp','Parch', 'Fare','Embarked']]

test_data = pd.read_csv('test.csv')
X_test = test_data[['Pclass', 'Sex', 'Age','SibSp','Parch', 'Fare','Embarked']]

#Missing data
imputer = TitanicImputer()
imputer = imputer.fit(X_train['Embarked'])
X_train['Embarked'] = imputer.transform(X_train['Embarked'])

imputer = imputer.fit(X_test['Embarked'])
X_test['Embarked'] = imputer.transform(X_test['Embarked'])

imputer = Imputer(missing_values = 'NaN', strategy='mean', axis = 0)
imputer = imputer.fit(X_train[['Age']])
X_train['Age'] = imputer.transform(X_train[['Age']])
imputer = imputer.fit(X_test[['Age']])
X_test['Age'] = imputer.transform(X_test[['Age']])

imputer = Imputer(missing_values = 'NaN', strategy='median', axis = 0)
imputer = imputer.fit(X_train[['Fare']])
X_train['Fare'] = imputer.transform(X_train[['Fare']])
imputer = imputer.fit(X_test[['Fare']])
X_test['Fare'] = imputer.transform(X_test[['Fare']])

#Categorical data
label_encoder = LabelEncoder()
X_train['Sex'] = label_encoder.fit_transform(X_train['Sex'])
X_test['Sex'] = label_encoder.transform(X_test['Sex'])
X_train['Embarked'] = label_encoder.fit_transform(X_train['Embarked'])
X_test['Embarked'] = label_encoder.transform(X_test['Embarked'])


encoder = OneHotEncoder(categorical_features=[1])
X_train = encoder.fit_transform(X_train).toarray()
X_test = encoder.transform(X_test).toarray()
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

encoder = OneHotEncoder(categorical_features=[6])
X_train = encoder.fit_transform(X_train).toarray()
X_test = encoder.transform(X_test).toarray()
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

#Machine Learning Algorithms for classification
"""
Logistic Regression -> This is linear we do not do it
K-Nearest Neighbor
Support Vector Machine (SVM)
Kernel SVM
Naive Bayes
Decision Tree Classification
Random Forest
"""
#K-Nearest Neighbor; It needs feature scaling

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors= 5,
                                  metric='minkowski',
                                  p=2)
classifier.fit(X_train, y_train)

y_pred_kneighbors = classifier.predict(X_test)