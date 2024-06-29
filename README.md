Importing libraries:
import pandas as pd ##(pandas library)
import numpy as np ##(adding numpy to work with arrays)
import matplotlib.pyplot as plt ##(adding visualisation libraries)
%matplotlib inline
import seaborn as sns
Importing data to notebook:
df=pd.read_csv("heart.csv")
df
Checking the shape (rows & columns) of data:
df.shape
Checking the information of data:
df.info()
Finding null values in data:
df.isnull().sum()
Finding value count of different columns:
df["sex"].value_counts()
Visualisation of column data using graph:
sns.countplot(x="sex",data=df)
Dropping unnecessary columns:
X=df.drop("target",axis=1)
X
Defining Y
Y=df["target"]
Checking the type of dependent column:
type(Y)
Changing dependent value to frame:
Y=Y.to_frame()
Y
Importing sklearn libraries (used for train-test split & for Accuracy):
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
Testing & training model:
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
print("shape of X_train : ",X_train.shape)
print("shape of X_test : ",X_test.shape)
print("shape of Y_train : ",Y_train.shape)
print("shape of Y_test : ",Y_test.shape)
Accuracy using Linear Regression:
regressor=LinearRegression()
regressor.fit(X,Y)
X_test_pred=regressor.predict(X_test)
X_test_pred
r2_train=metrics.r2_score(Y_test,X_test_pred)
print("Accuracy : ",r2_train*100)
Accuracy using Random Forest Classifier:
model=RandomForestClassifier()
model.fit(X_train,Y_train)
X_test_pred=model.predict(X_test)
accuracy=accuracy_score(X_test_pred,Y_test)
print("Accuracy :", accuracy*100,"%")
Accuracy using SVC:
from sklearn.svm import SVC
clf=SVC()
clf.fit(X_test,Y_test)
accu=clf.score(X_test,Y_test)
print("Accuracy : ", accuracy*100,"%")
