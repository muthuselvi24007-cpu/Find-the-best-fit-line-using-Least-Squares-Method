# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [20, 25, 35, 45, 50, 60, 65, 70, 80, 85]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)
X = df[['Hours']]
y = df['Marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nActual vs Predicted Marks:")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Hours vs Marks (Simple Linear Regression)')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.legend()
plt.show()
hours = float(input("\nEnter study hours to predict marks: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for  {hours} hours of study = {predicted_marks[0]:.2f}")
```

## Output:
<img width="1166" height="819" alt="image" src="https://github.com/user-attachments/assets/9132f63f-8753-466b-8eb7-207660256f76" />
<img width="1543" height="818" alt="image" src="https://github.com/user-attachments/assets/565342e9-8379-43ca-8cb4-00a68e3fac2c" />



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
