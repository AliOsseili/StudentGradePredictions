import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pylot
import pickle
from matplotlib import style
#Imports

data = pd.read_csv("student-mat.csv", sep=";")
#Load data into a panda model

data = data[["G1", "G2", "G3", "studytime", "failures","absences",]]
#grab the data we are interested in

#print(data.head())
#print out the top 5 rows, uncomment line 13 to see this

predict = "G3"
#define what we are predicting
x = np.array(data.drop([predict], axis='columns'))
y = np.array(data[predict])
#setup our axis

x_train, x_test ,y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
#split the train data and test data, so we train our machine on 10 percent of all data in the set, the rest is predicted

"""linear = linear_model.LinearRegression()
#choosing the model
linear.fit(x_train,y_train)
#fitting the model (actually doing the predicting and storing the results)
acc = linear.score(x_test,y_test)
#grab the accuracy from the model
print(acc)
#print the accuracy out

with open("studentmodel.pickle","wb") as f:
    pickle.dump(linear,f)"""
#the above section is for if you wish to generate a new model. You can uncomment if you wish to do so.

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test,y_test)
#grab the accuracy from the model
print(acc)
#print the accuracy out

print("Co: \n", linear.coef_)
#print out our coefficients on our linear model
print("Intercept: \n ", linear.intercept_)
#print out the intercept

#visual viewing of predictions below
#predictions = linear.predict(x_test)
#for x in range(len(predictions)):
#    print(predictions[x], x_test[x],y_test[x])