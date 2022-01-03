import numpy as np
import warnings #warnings to avoid using lower k value than number of groups
from math import sqrt
from collections import Counter
from numpy.core.numeric import full
import pandas as pd
import random 

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = [] #list to store another list with first element as the distance
    # and second element as the class (color)

    for group in data: #group is a key/value object
        for features in data[group]: #features are one of the values, which are the coordinates
            #calculate distance as vectors
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance,group]) #add to list of distances
    
    votes = [i[1] for i in sorted(distances)[:k]] #take k closest elements
    vote_result = Counter(votes).most_common(1)[0][0] #find the most commonly occuring class

    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data') #read data
df.replace('?', -99999, inplace=True) #replace missing data as outliers
#id doesn't have any relevance to the tumor, so get rid of it
df.drop(['id'], 1, inplace=True) #remove the id column
full_data = df.astype(float).values.tolist() #convert to floats
random.shuffle(full_data) #shuffle data

#basically do the same things as in KNN.py
test_size = 0.2
train_set = {2:[], 4:[]} #create dictionaries to store data
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))] #grab data up to test_size*len of full_data
test_data = full_data[-int(test_size*len(full_data)):] #grap data after test_size*len of full_data

for i in train_data: #populate dictionaries
    train_set[i[-1]].append(i[:-1]) 
    #train_set[i[-1]] is the class, 2 or 4 for malignant
    #append(i[:-1]) appends the data up to the class, which is the last element

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set: #call the function for each set in test_set, using the data as the prediction value
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5) #train_set is used as the data to classify the values from test_set
        if group == vote:
            correct += 1 #store for accuracy calculation
        total += 1

print('Accuracy: ', correct/total) 