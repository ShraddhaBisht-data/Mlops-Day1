import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("C:\Mlops Day 1\Data\data.csv")


# preprocessing
#....


# Train-test split
X,y = df[["TV", "Radio", "Newspaper"]],df[["Sales"]]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=67)

# Linear Regression model
model = LinearRegression()
model.fit(Xtrain, ytrain)


ypred = model.predict(Xtest)
r2 = r2_score(ytest, ypred)
rmse = root_mean_squared_error(ytest, ypred)

# Model dump
joblib.dump(model,r"C:\Mlops Day 1\Models\linear_reg_model.pkl") 
