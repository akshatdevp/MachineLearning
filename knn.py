import numpy as np
from sklearn import preprocessing,model_selection,neighbors
import pandas as pd
import matplotlib as plt
df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)#to make it a huge outlier
df.drop(['id'],1,inplace=True)# id is useless for classification, and causes huge chaos since knn doesn't handle it well
x = np.array(df.drop(['class'],1))
y = np.array(df['class'])#we need to predict class
x_train, x_test, y_train, y_test=model_selection.train_test_split(x,y,test_size=0.2)
clf =neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
acc=clf.score(x_test,y_test)
print(acc)
example_values=np.array([[4,2,1,1,1,1,2,1,1]])#need 2d array 
example_values=example_values.reshape(len(example_values),-1)#
prediction=clf.predict(example_values)
print(prediction)
