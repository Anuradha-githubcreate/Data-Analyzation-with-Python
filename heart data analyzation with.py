import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df=pd.read_csv("heart.csv")
df
df.shape
df.info()
df.isnull().sum()
df["sex"].value_counts()
sns.countplot(x="sex",data=df)
plt.show()
df["target"].value_counts()
sns.countplot(x="target",data=df)
plt.show()
df["chol"].value_counts()
sns.countplot(x="chol",data=df)
plt.show()
X=df.drop("target",axis=1)
X
Y=df["target"]
type(Y)
Y=Y.to_frame()
Y
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
print("shape of X_train : ",X_train.shape)
print("shape of X_test : ",X_test.shape)
print("shape of Y_train : ",Y_train.shape)
print("shape of Y_test : ",Y_test.shape)
regressor=LinearRegression()
regressor.fit(X,Y)
X_test_pred=regressor.predict(X_test)
X_test_pred
r2_train=metrics.r2_score(Y_test,X_test_pred)
print("Accuracy : ",r2_train*100)
model=RandomForestClassifier()
model.fit(X_train,Y_train)
X_test_pred=model.predict(X_test)
accuracy=accuracy_score(X_test_pred,Y_test)
print("Accuracy :", accuracy*100,"%")
from sklearn.svm import SVC
clf=SVC()
clf.fit(X_test,Y_test)
accu=clf.score(X_test,Y_test)
print("Accuracy : ", accuracy*100,"%")