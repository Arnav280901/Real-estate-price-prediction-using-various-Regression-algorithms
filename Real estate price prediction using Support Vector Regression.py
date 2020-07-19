# REAL ESTATE PRICE PREDICTION USING SUPPORT VECTOR REGRESSION

# IMPORTING THE LIBRARIES
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("datasets_88705_204267_Real estate.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# EXPLORATORY DATA ANALYSIS
num_bins = 10
plt.hist(dataset['Y house price of unit area'],
         num_bins, facecolor='blue', edgecolor='black')
plt.title('Histogram of House Prices')
plt.xlabel('House Prices')
plt.ylabel('Number of Houses')
plt.show()

# FEATURE SCALING
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# SPLITTING THE DATASET INTO TRAINING SET AND TEST SET
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# TRAINING THE SVR MODEL ON THE WHOLE DATASET
regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train)

# PREDICTING TEST SET RESULTS
print("[ PREDICTED PRICES, TEST SET PRICES ]")
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
y_test = sc_y.inverse_transform(y_test)
y_pred = sc_y.inverse_transform(y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# VISUALIZING THE ACCURACY OF THE MODEL
plt.scatter(y_test.reshape(len(y_test), 1),
            y_pred.reshape(len(y_pred), 1), color='red')
plt.title('REAL ESTATE PRICE PREDICTION (Support Vector Regression)')
plt.xlabel('ACTUAL PRICE')
plt.ylabel('PREDICTED PRICE')
plt.show()
