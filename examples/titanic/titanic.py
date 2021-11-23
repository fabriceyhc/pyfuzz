#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/29/21 6:37 AM
# @Author  : Jiyuan Wang
# @File    : titanic.py
# jteoh: modifications to functionalize commands and utilize SVC
# fhc: modifications to fix relative paths to datasets

import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def load_data():
    script_dir = os.path.dirname(__file__)
    train = pd.read_csv(os.path.join(script_dir, 'data/train.csv'))
    test  = pd.read_csv(os.path.join(script_dir, 'data/test.csv'))

    train_test_data = [train, test]  # combining train and test dataset

    for dataset in train_test_data:
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                     "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                     "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}
    for dataset in train_test_data:
        dataset['Title'] = dataset['Title'].map(title_mapping)

    train.drop('Name', axis=1, inplace=True)
    test.drop('Name', axis=1, inplace=True)

    sex_mapping = {"male": 0, "female": 1}
    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping)

    # fill missing age with median age for each title (Mr, Mrs, Miss, Others)
    train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
    test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

    train.groupby("Title")["Age"].transform("median")
    for dataset in train_test_data:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
        dataset.loc[dataset['Age'] > 62, 'Age'] = 4

    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

    for dataset in train_test_data:
        dataset['Cabin'] = dataset['Cabin'].str[:1]
    cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
    for dataset in train_test_data:
        dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    # fill missing Fare with median fare for each Pclass
    train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

    # jteoh: Added in fill NA command for cabin from notebook
    # fill missing Fare with median fare for each Pclass
    train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
    test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


    train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
    test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
    family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
    for dataset in train_test_data:
        dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)



    features_drop = ['Ticket', 'SibSp', 'Parch']
    train = train.drop(features_drop, axis=1)
    test = test.drop(features_drop, axis=1)
    train = train.drop(['PassengerId'], axis=1)
    train_data = train.drop('Survived', axis=1)
    target = train['Survived']


    #ML: kNN, I delete this part
    #k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    #clf = KNeighborsClassifier(n_neighbors=13)
    #scoring = 'accuracy'
    #score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    #print(score)

    return (train_data, test, target)


# jteoh: I added this for the sake of having some actual computation component
def train_model_KNeighbors(train_data, target):
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = KNeighborsClassifier(n_neighbors = 13)
    scoring = 'accuracy'
    score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(score)


def train_SVC(train_data, target):
    global clf
    clf = SVC()
    clf.fit(train_data, target)
    return clf

def setup():
    (train_data, test, target) = load_data()
    # train_model_KNeighbors(train_data, target)
    model = train_SVC(train_data, target)
    return (train_data, test, target, model)


def predict(input):
    prediction = clf.predict(input)
    # print(prediction)
    return prediction

# https://stackoverflow.com/questions/45504241/create-single-row-python-pandas-dataframe
def build_test_input(pclass, sex, age, fare, cabin, embarked, title, family_size):
    test_input = pd.DataFrame(columns=['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked', 'Title', 'FamilySize'])
    test_input.loc[0] = [pclass, sex, age, fare, cabin, embarked, title, family_size]
    return test_input

if __name__ == '__main__':
    (train_data, test, target, model) = setup()

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = model.predict(test_data)


