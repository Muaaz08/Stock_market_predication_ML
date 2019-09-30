from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"C:\Users\inas\Desktop\Machine_learning_project\beancoder_examples-master\linear_reg_article\google.csv")
x = np.array(data["Date"])
x = np.reshape(x, (len(x), 1))

y = np.array(data["Open"])
y = np.reshape(y, (len(y), 1))

colors = ['r', 'g', 'b', 'y', 'k', 'm', 'c']
plt.plot(x,y, 'b*')

for n in range(1,10):
    
    polyfit = PolynomialFeatures(degree=n)
    xn = polyfit.fit_transform(x)
    print(xn)
    model = linear_model.LinearRegression()
    model.fit(xn, y)
    y1 = model.predict(xn)
    error = mse(y, y1)
    print('degree=', n, "error=", error)
    plt.plot(x, y1, colors[(n-1)%7])

plt.show()

