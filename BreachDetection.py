# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 12:11:07 2018

@author: Ishan Shah
"""

#Step 1: Import the libraries

# machine learning classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
# For data manipulation
import numpy as np

# To plot
import matplotlib.pyplot as plt
import seaborn

#Step 2: Fetch data

import fix_yahoo_finance as yf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict
fundCompanies = {'ADRIATICA CAPITAL d.o.o.' : 1, 'ADRIATICA CAPITAL d.o.o.52':2, 'AGRAM Invest d.d.':3,
                 'ALFA INVEST d.o.o12':4, 'Allianz Invest d.o.o.':5, 'ALTERNATIVE INVEST d.o.o.':6,
                 'AUCTOR INVEST d.o.o.47':7, 'CAIB INVEST d.o.o.22':8, 'Erste Asset Management d.o.o.75':9,
                 'ERSTE - INVEST d.o.o.':10, 'FIMA GLOBAL INVEST d.o.o.':11, 'HPB-INVEST d.o.o.':12,
                 'HYPO-ALPE-ADRIA INVEST d.d.':13, 'ICAM d.o.o.':14, 'ICF INVEST d.o.o.34':15,
                 'ILIRIKA INVESTMENTS d.o.o.':16, 'INTERINVEST d.o.o.':17, 'KD INVESTMENTS  d.o.o.':18,
                 'LOCUSTA INVEST d.o.o.':19, 'MP INVEST d.d.37':20, 'NFD Aureus Invest d.d.':21,
                 'NFD Aureus Invest d.d.26,37':22, 'NFD Aureus Invest d.d.26':23, 'NFD Aureus Invest d.d.':24,
                 'OTP INVEST d.o.o.':25, 'PLATINUM INVEST d.o.o.':26, 'PROSPECTUS INVEST d.o.o.33':27,
                 'ST INVEST d.o.o.':28, 'ST INVEST d.o.o.54,55,56':29, 'VB INVEST d.o.o.':30,
                 'ZB INVEST d.o.o.':31}
types = {'M':1, 'N':2, 'O':3, 'D':4}
offerings = {'JP':1, 'PP':2}
def toType(c) :
    if (c in types.keys()) :
        return types[c]
    else :
        return 0
def toOffering(c) :
    if (c in offerings.keys()) :
        return offerings[c]
    else :
        return 0
def toFC(c) :
    if (c in fundCompanies.keys()) :
        return fundCompanies[c]
    else :
        return 0
def getData(file) :
    my_sheet_name = 'OIF NAV 2013'
    converters = {'Type': toType, 'Offering':toOffering,'FundCompany':toFC}
    Df = pd.read_excel(file, sheet_name=my_sheet_name, converters=converters)
    print(Df.head())  # shows headers with top 5 rows
    Df = Df.dropna()
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    for column_name in Df.columns:
        if Df[column_name].dtype == object:
            Df[column_name] = le.fit_transform(Df[column_name])
        else:
            pass



    # Step 3: Determine the target variable
    y = Df.iloc[:,-1]

    # Step 4: Creation of predictors variables

    X = Df.iloc[:, 0:-1]

    return X, y

X_train, y_train = getData("./data/funds1.xlsx")

X_test, y_test = getData("./data/funds2.xlsx")



#Step 6: Create the machine learning classification model using the train dataset

cls = SVC().fit(X_train, y_train)

#Step 7: The classification model accuracy

accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))
print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))

#Step 8: Prediction

 #cls.predict(X)


