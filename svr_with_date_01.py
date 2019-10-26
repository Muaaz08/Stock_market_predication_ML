import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR

date = []
price = []

def get_data(filename):
    with open(filename, 'r')as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            date.append(int(row[0].split('-')[1]))
            price.append(float(row[1]))
        return

def predict_price(date,price,x):
    date = np.reshape(date,(len(date),1))

    svr_lin = SVR(kernel='linear',C=1e3)
    svr_poly = SVR(kernel='poly',C=1e3,degree = 2)
    svr_rbf = SVR(kernel='rbf',C=1e3)
    svr_lin.fit(date,price)
    svr_poly.fit(date,price)
    svr_rbf.fit(date,price)
    
    plt.scatter(date,price,color='black',label='Data')
    plt.plot(date,svr_lin.predict(date),color='red',label='Linear model')
    plt.plot(date,svr_poly.predict(date),color='green',label='Polynomial model')
    plt.plot(date,svr_rbf.predict(date),color='blue',label='RBF model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Market Prediction')
    plt.legend()
    plt.show()
    return svr_lin.predict(x)[0],svr_rbf.predict(x)[0],svr_poly.predict(x)[0]

get_data('daily.csv')
pp = predict_price(date,price,[[31]])

print(pp)




