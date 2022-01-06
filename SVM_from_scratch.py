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
        self.min_feature_value = min(all_data)
        all_data = None
        
        #start with big steps, will progressively get smaller, costly performance to add more smaller steps
        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]
        #support vectors yi*(xi.w + b) = 1 when optimized
        #not as important for b to be very precise for the amount of computation required
        b_range_multiple = 5
        b_multiple = 2

        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            #this can be done because it is convex
            optimized = False
            while not optimized:
                #np.arange() gives the ability to set start and stop points and step sizes
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple), self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation #multiply w_t by each transform
                        found_option = True
                        for i in self.data: #weakest link in the SVM, all the data needs to be loaded
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                        
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
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s = 200, marker="*", c=self.colors[classification])

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #hyperplane = x.w+b
        def hyperplane(x,w,b,v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        #positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max], [psv1,psv2], 'k')

        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max], [nsv1,nsv2], 'k')

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max], [db1,db2], 'y--')

        plt.show()

#create dictionary with -1 and +1 classes, each storing arrays of arrays
data_dict = {-1:np.array([[1,7], [2,8], [3,8]]), 1:np.array([[5,1], [6,-1], [7,3]])} 

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

#datapoints to classify
predict_us = [[0,10], [1,3], [3,4], [3,5], [5,5], [5,6], [6,-5], [5,8]]
for p in predict_us:
    svm.predict(p)

svm.visualize()