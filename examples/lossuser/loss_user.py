#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/9/21 2:41 PM
# @Author  : Jiyuan Wang
# @File    : loss_user.py

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import os
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
#使用PCA方法对数据进行降维
from sklearn.decomposition import PCA
#导入管道机制进行流水作业
from sklearn.pipeline import Pipeline
#导入自带的评估模型准确率的函数
from sklearn.metrics import accuracy_score
#导入分类算法SVC， 其它还有NuSVC，和LinearSVC 。另一类是回归算法库，包括SVR， NuSVR，和LinearSVR
from sklearn.svm import SVC
#决策树分类器
from sklearn.tree import DecisionTreeClassifier
#随机森林分类器
from sklearn.ensemble import RandomForestClassifier
#KNN分类器
from sklearn.neighbors import KNeighborsClassifier
#Adaboost分类器
from sklearn.ensemble import AdaBoostClassifier

def load_data():
    script_dir = os.path.dirname(__file__)
    customerDf = pd.read_csv(os.path.join(script_dir, 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    #customerDf[['TotalCharges']].astype(float)
    customerDf['TotalCharges'] = pd.to_numeric(customerDf['TotalCharges'], errors="coerce")

    customerDf.loc[:, 'TotalCharges'].fillna(customerDf['TotalCharges'].mean(), inplace=True)
    customerDf.loc[:, 'tenure'].replace(to_replace=0, value=1, inplace=True)

    customerID = customerDf['customerID']
    customerDf.drop(['customerID'], axis=1, inplace=True)

    cateCols = [c for c in customerDf.columns if customerDf[c].dtype == 'object' or c == 'SeniorCitizen']
    global dfCate
    dfCate = customerDf[cateCols].copy()
    dfCate.head(3)

    for col in cateCols:
        if dfCate[col].nunique() == 2:
            dfCate[col] = pd.factorize(dfCate[col])[0]
        else:
            dfCate = pd.get_dummies(dfCate, columns=[col])
    dfCate['tenure'] = customerDf[['tenure']]
    dfCate['MonthlyCharges'] = customerDf[['MonthlyCharges']]
    dfCate['TotalCharges'] = customerDf[['TotalCharges']]

    dropFea = ['gender', 'PhoneService',
               'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
               'DeviceProtection_No internet service', 'TechSupport_No internet service',
               'StreamingTV_No internet service', 'StreamingMovies_No internet service',
                'OnlineSecurity_No', 'OnlineBackup_No',
                'DeviceProtection_No','TechSupport_No',
                'StreamingTV_No', 'StreamingMovies_No',
                'MultipleLines_No', 'Contract_Month-to-month',
                'StreamingMovies_Yes', 'StreamingTV_Yes',
                'PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check',
                'InternetService_Fiber optic','MultipleLines_No phone service',
                'PaymentMethod_Mailed check', 'PaymentMethod_Bank transfer (automatic)',
                'OnlineSecurity_Yes', 'InternetService_DSL','InternetService_No',
                'OnlineBackup_Yes','DeviceProtection_Yes','PaperlessBilling'
               ]
    dfCate.drop(dropFea, inplace=True, axis=1)
    # 最后一列是作为标识
    target = dfCate['Churn'].values
    # 列表：特征和1个标识
    columns = dfCate.columns.tolist()

    # 列表：特征
    columns.remove('Churn')
    # 含有特征的DataFrame
    features = dfCate[columns].values
    # 30% 作为测试集，其余作为训练集
    # random_state = 1表示重复试验随机得到的数据集始终不变
    # stratify = target 表示按标识的类别，作为训练数据集、测试数据集内部的分配比例
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify=target,
                                                        random_state=1)

    return train_x, test_x, train_y, test_y

def train_model(train_x, train_y):
    classifiers = [
        SVC(random_state=1, kernel='rbf'),
        DecisionTreeClassifier(random_state=1, criterion='gini'),
        RandomForestClassifier(random_state=1, criterion='gini'),
        KNeighborsClassifier(metric='minkowski'),
        AdaBoostClassifier(random_state=1),
    ]
    # 分类器名称
    classifier_names = [
        'svc',
        'decisiontreeclassifier',
        'randomforestclassifier',
        'kneighborsclassifier',
        'adaboostclassifier',
    ]
    # 分类器参数
    # 注意分类器的参数，字典键的格式，GridSearchCV对调优的参数格式是"分类器名"+"__"+"参数名"
    classifier_param_grid = [
        {'svc__C': [0.1], 'svc__gamma': [0.01]},
        {'decisiontreeclassifier__max_depth': [6, 9, 11]},
        {'randomforestclassifier__n_estimators': range(1, 11)},
        {'kneighborsclassifier__n_neighbors': [4, 6, 8]},
        {'adaboostclassifier__n_estimators': [70, 80, 90]}
    ]

    global s_model
    s_model = AdaBoostClassifier(n_estimators=80)
    s_model.fit(train_x, train_y)

    return s_model

def run_model(pred_x):
    pred_y = s_model.predict(pred_x)
    predDf = pd.DataFrame({'Churn': pred_y})

def set_up():
    train_x, test_x, train_y, test_y = load_data()
    model = train_model(train_x,train_y)
    return (test_x,test_y,model)

def build_test_input(SeniorCitizen,Partner,Dependents,MultipleLines_Yes,TechSupport_Yes,
                                   Contract_Oneyear,Contract_Twoyear,tenure,MonthlyCharges,TotalCharges):
    test_input = pd.DataFrame(
        columns=['SeniorCitizen','Partner','Dependents','MultipleLines_Yes','TechSupport_Yes',
                                   'Contract_Oneyear','Contract_Twoyear','tenure','MonthlyCharges','TotalCharges'])
    test_input.loc[0] = [SeniorCitizen,Partner,Dependents,MultipleLines_Yes,TechSupport_Yes,
                                   Contract_Oneyear,Contract_Twoyear,tenure,MonthlyCharges,TotalCharges]
    return test_input

if __name__ == '__main__':
    test_x, test_y, model = set_up()
    pd.set_option('max_colwidth',100)
    pd.set_option('display.max_columns',None)

    pred_x = dfCate.drop(['Churn'],axis=1).tail(10)
    print(pred_x)
    run_model(pred_x)





