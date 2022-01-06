import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True) : #visualization will give option to enable/disable plots
        self.visualization = visualization #set value to whatever the user input or True by default
        self.colors = {1:'r', -1:'b'} #class 1 will be red, class -1 will be blue
        if self.visualization: #plot figure
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    def fit(self, data): #fit means train, this is a convex optimization problem
        self.data = data
        #{ ||w||: [w,b] }
        opt_dict = {} #key is magnitude of w, values are w and b

        #transforms will be applied to vector w each step
        transforms = [[1,1], [-1,-1], [-1,1], [1,-1]]

        all_data = [] #want to be able to find max and min values
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_featur_value = min(all_data)
        all_data = None
        
        #start with big steps, will progressively get smaller, costly performance to add more smaller steps
        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]
        #support vectors yi*(xi.w + b) = 1 when optimized
        #not as important for b to be very precise for the amount of computation required
        b_range_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            #this can be done because it is convex
            optimized = False
            while not optimized:
                #np.arange() gives the ability to set start and stop points and step sizes
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple), self.max_feature_value * b_range_multiple, step * b_range_multiple):
                    for transformation in transforms:
                        w_t = w * transformation #multiply w_t by each transform
                        found_option = True
                        for i in self.data: #weakest link in the SVM, all the data needs to be loaded
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    #break
                        
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else: 
                    w = w - step #step

            norms = sorted([n for n in opt_dict]) #sorted list of magnitudes
            opt_choice = opt_dict[norms[0]] #vector at the smallest magnitude in opt_dict

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2


    def predict(self, features): #use the sign of the equation below to determine if the data is -1 or +1 class
        #sign( x.w + b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b) #use np.dot for dot product, np.sign for sign of equation

        return classification


#create dictionary with -1 and +1 classes, each storing arrays of arrays
data_dict = {-1:np.array([[1,7], [2,8], [3,8]]), 1:np.array([[5,1], [6,-1], [7,3]])} 
