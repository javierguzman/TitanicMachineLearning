# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import Imputer
from titanic_imputer import TitanicImputer

train_data = pd.read_csv('train.csv')

y_train = train_data['Survived'].values.reshape(len(train_data), 1)
X_train = train_data[['Pclass', 'Sex', 'Age','SibSp','Parch', 'Fare','Embarked']]

imputer = TitanicImputer()
imputer = imputer.fit(X_train['Embarked'])
X_train['Embarked'] = imputer.transform(X_train['Embarked'])


imputer = Imputer(missing_values = 'NaN', strategy='mean', axis = 0)
imputer = imputer.fit(X_train[['Age']])
X_train['Age'] = imputer.transform(X_train[['Age']])

