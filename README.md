# EX 09: Implementation-of-SVM-For-Spam-Mail-Detection



## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Character Encoding Detection:

* You’ve used the chardet library to detect the character encoding of the file “spam.csv”.

* The detected encoding is likely to be Windows-1252 (also known as cp1252).
Data Loading and Exploration:

* You’ve loaded the dataset from “spam.csv” using pandas and specified the encoding as Windows-1252.

* You’ve printed the first five rows of the dataset using data.head().

* Additionally, you’ve displayed information about the dataset using data.info().

2. Data Preprocessing:

* You’ve split the data into training and testing sets using train_test_split.

* You’ve used CountVectorizer to convert text data (in column “v2”) into numerical features for SVM training.

3. Model Training and Prediction:

* You’ve initialized an SVM classifier (svc) and trained it on the training data.

* You’ve predicted the labels for the test data using y_pred.

4. Model Evaluation:

* You’ve calculated the accuracy of the model using metrics.accuracy_score.

## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rakesh V
RegisterNumber:  212222110036
*/
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
print("The First five Data:\n")
print(data.head())
print("\nThe Information:\n")
print(data.info())
print("\nTo count the Number of null values in dataset:\n")
print(data.isnull().sum())
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("\nThe Y_prediction\n")
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("\nAccuracy:\n")
print(accuracy)

```

## Output:

![image](https://github.com/rakeshcoder2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121490890/0ac845f5-e0e9-45c5-9c26-6fb8039e7f33)

![image](https://github.com/rakeshcoder2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121490890/f9cb5ec3-fa00-4cfc-aeec-9cbbdb847ef8)



![image](https://github.com/rakeshcoder2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121490890/a1236852-301a-4e49-852d-436c27959c1a)



![image](https://github.com/rakeshcoder2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121490890/2c8f2e2f-1a31-4348-a67b-a1d98c7da305)
![image](https://github.com/rakeshcoder2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121490890/da943844-7881-4969-b723-881d7a0aa5c6)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
