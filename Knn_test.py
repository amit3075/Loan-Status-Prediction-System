
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from KNearestNeighbors import KNearestNeighbors

data = pd.read_csv('loan.csv')
print(data.head())
#label encoding for Loan_Status
data.Loan_Status = data.Loan_Status.map({'Y':1,'N':0})
## Label encoding for gender
data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()
data.Married=data.Married.map({'Yes':1,'No':0})
## Labelling 0 & 1 for Dependents
data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
## Labelling 0 & 1 for Education Status
data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})
## Labelling 0 & 1 for Employment status
data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})
## Labelling 0 & 1 for Property area
data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
#time to fill the mising values
data.Credit_History.fillna(np.random.randint(0,2),inplace=True)
data.Married.fillna(np.random.randint(0,2),inplace=True)
## Filling with median
data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)

## Filling with mean
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)
## Filling Gender with random number between 0-2
# from random import randint
data.Gender.fillna(np.random.randint(0,2),inplace=True)
## Filling Dependents with median
data.Dependents.fillna(data.Dependents.median(),inplace=True)
data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)

X = data.iloc[:,6:10].values
y = data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=30)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# import random
# k_min = 1
# k_max = 10
# k = random.randint(1, 100)
# k = k if k % 2 != 0 else k + 1
# knn =KNearestNeighbors(k = random.randint(k_min, k_max))
# knn = KNearestNeighbors(k = k if k % 2 != 0 else k + 1)
knn = KNearestNeighbors(k=5)
knn.fit(X_train, y_train)


# knn.predict(np.array([60,100,1000,20]).reshape(1,4))

Applicant_income = int(input("Enter the applicant_income"))
Coapplicant_income = int(input("Enter the coapplicant_income"))
Loan_amount = int(input("Enter the loan_amount"))
Loan_amount_term = int(input("Enter the loan amount term in day"))
X_test_list= np.array([[Applicant_income], [Coapplicant_income], [Loan_amount], [Loan_amount_term]]).reshape(1, 4)
# if float(int(Applicant_income)/2) < float(Loan_amount):
#    result = knn.predict(X_test_list)
#    result ==0
#    print("will be deined the loan")

# reuslt=knn.predict(X_test)
# print(reuslt)
# pred = knn.predict(X_test)
# print(pred)


pred2 =knn.predict(X_test_list)
print(pred2)


from sklearn.metrics import confusion_matrix
pred1 =knn.predict(X_test)

cm = confusion_matrix(y_test,pred1)
print(cm)


from sklearn.metrics import classification_report
print(classification_report(y_test,pred1))

print("accuracy_pred:{}".format(accuracy_score(y_test,pred1)))
knnaccu =accuracy_score(y_test,pred1)
print(knnaccu)

import joblib
from joblib import dump
dump(knn, './model.joblib')
mp =knn.predict(X_test_list)