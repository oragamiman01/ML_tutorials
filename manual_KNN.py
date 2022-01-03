import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings #warnings to avoid using lower k value than number of groups
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')


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

#dataset is a dictionary w/ colors as keys and coordinates as values
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]} #k is black, r is red
new_features = [5,7] #data we want to classify

#create graph
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i) 
plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features) #make the classification
print(result)
plt.scatter(new_features[0], new_features[1], s=100, color = result) #plot resulting color

plt.show()
