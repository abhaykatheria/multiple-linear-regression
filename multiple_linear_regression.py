# Multiple Linear Regression
#@author-Abhay Katheria

#computing cost function
def Cost(X,y,theta):
        tobesummed = np.power(((X @ theta.T)-y),2)
        return np.sum(tobesummed)/(2 * len(X))


def regressor(X_train,y_train):
        theta = np.zeros([1,len(X[1,:])])
        #setting hyper parameters
        alpha = 0.01
        iters = 1000
        #the cost array has been made to store value of error after ech iteration
        cost = np.zeros(iters)
        for i in range(iters):
            #the holy grail :GRADIENT DESCENT 
            theta = theta - (alpha/len(X_train)) * np.sum(X_train * (X_train @ theta.T - y_train), axis=0)
            cost[i] = Cost(X_train, y_train, theta)#STORING THE ERROR
        #plotting the error vs iteration    
        fig, ax = plt.subplots()    
        ax.plot(np.arange(iters), cost, 'r')  
        ax.set_xlabel('Iterations')  
        ax.set_ylabel('Cost')  
        ax.set_title('Error vs. iterations')     
        return theta

#mporting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#reading the dataset 
dataset = pd.read_csv('50_Startups.csv')
#separating dependet and ndependet variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(X[1,:])].values
#this line was written because the array was not a numpy array of desired shape which was major bug
y = y.reshape(len(y),1)

# Encoding categorical data


#Here we have city as a categorical independant variable

# Encoding the Independent Variable
#SKLEARN offers awesome fuctioality to deal with categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#we did this to avoid dumy variable trap
X=X[:, 1:]

X=(X-X.mean())/X.std() #normalisation

#addding a column of ones to independent variable (x0)
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
#splitting train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 1)
#magic begins
theta = regressor(X_train,y_train)
#predicting the test 
pred = X_test @ theta.T