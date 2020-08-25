import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
df=pd.read_csv("iris.csv")

df1=pd.read_csv("iris.csv")

x=df[[ 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

y=df[['Species']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
lr=KNeighborsClassifier()
lr.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
import pickle

filename="iris.pkl"

pickle.dump(lr, open(filename, 'wb'))
