import pandas as pd
import  matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
data=pd.read_csv('carprices.csv')
x=data[['Mileage','Age(yrs)']]
y=data['Sell Price($)']
x_train,x_test,y_train,y_test=train_test_split(x,y)
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
train=model.predict(x_test)
accurate=model.score(x_test,y_test)
print(accurate)