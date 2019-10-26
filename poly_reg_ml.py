import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv(r"google.csv")
#print(df)

x = np.array(df['Date'])
x = np.reshape(x,(len(x),1))

y = np.array(df['Open'])
y = np.reshape(y,(len(y),1))

#Linear Regression
reg = linear_model.LinearRegression()
reg.fit(x,y)

#Polynomial Regression
poly = PolynomialFeatures(degree = 9)
x_poly = poly.fit_transform(x)
reg2 = linear_model.LinearRegression()
reg2.fit(x_poly,y)

#predict of linear regression
y1 = reg.predict(x)

#predict of poly regression
y2 = reg2.predict(x_poly)

plt.plot(x,y,'r*',label='Data points') #data points
plt.plot(x,y1,'b',label='Linear regression') #linear regression
plt.plot(x,y2,'y',label='Polynomial regression') #poly regression
plt.xlabel('Date')
plt.ylabel('Price(in US$)')
plt.title('Stock Price Prediction on Google Stocks')
plt.legend()
errorL = mse(y,y1)
errorP = mse(y,y2)

errorLp = 100*sum(abs((y-y1)/(max(y)-min(y))))/len(y)  #average percentage erro
errorPp = 100*sum(abs((y-y2)/(max(y)-min(y))))/len(y)  #average percentage erro


print("Actual price on",x[[0]],'is',y[[0]])
print("Prediction for price on",x[[0]],"from Lin",reg.predict([[26]])
      ,"and Poly",reg2.predict(poly.fit_transform([[26]])))
print('Mean Squared Error of linear',errorL,' & Poly',errorP)
print('Percentage Error of linear',errorLp,' & Poly',errorPp)

plt.show()




