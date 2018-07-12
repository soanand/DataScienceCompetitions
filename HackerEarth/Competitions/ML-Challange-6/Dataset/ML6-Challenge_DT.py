# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:58:50 2018

@author: soanand
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv("train.csv")

print(dataset.columns)

print(dataset.isnull().sum())

X = pd.DataFrame(dataset, columns=['area_assesed', 'district_id',
       'has_geotechnical_risk', 'has_geotechnical_risk_fault_crack',
       'has_geotechnical_risk_flood', 'has_geotechnical_risk_land_settlement',
       'has_geotechnical_risk_landslide', 'has_geotechnical_risk_liquefaction',
       'has_geotechnical_risk_other', 'has_geotechnical_risk_rock_fall',
       'has_repair_started', 'vdcmun_id'])

y = pd.DataFrame(dataset['damage_grade'])
print(X.head())

# Data Preprocessing
X['has_repair_started'] = X['has_repair_started'].fillna(0)
print(X.isnull().sum())

from sklearn.preprocessing import LabelEncoder
leAreaAssesed = LabelEncoder()
leAreaAssesed.fit(X['area_assesed'])
X['area_assesed'] = leAreaAssesed.transform(X['area_assesed'])
print(X.head())

print(y.head())
leDamageGrade = LabelEncoder()
leDamageGrade.fit(y['damage_grade'])
y['damage_grade'] = leDamageGrade.transform(y['damage_grade'])
print(y.head())

# Divide the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating model using decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predict the test result
y_pred = classifier.predict(X_test)


# Check the test accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

