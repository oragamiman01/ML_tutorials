import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data') #read data
df.replace('?', -99999, inplace=True) #replace missing data as outliers
#id doesn't have any relevance to the tumor, so get rid of it
df.drop(['id'], 1, inplace=True) #remove the id column

#features
X = np.array(df.drop(['class'],1))
#label (if the tumor is malignant or not)
y = np.array(df['class'])

#shuffle data into training and testing chunks
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

#classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train) #fit data to classifier

accuracy = clf.score(X_test, y_test) #accuracy, not confidence
print(accuracy)

#here we make two arrays of results that are fed into predict() to predict the outcome
#we make the array a list of lists so that taking the length will give the number of lists of results
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]]) 
#this way more results can be added in to make more predictions
example_measures =  example_measures.reshape(len(example_measures),-1) 

prediction = clf.predict(example_measures)
print(prediction)