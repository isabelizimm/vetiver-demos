import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import numpy as np
from joblib import dump

np.random.seed(500)


raw = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv')

df = pd.DataFrame(raw)
coffee = df[["total_cup_points", "aroma", "flavor", "sweetness", "acidity", \
    "body", "uniformity", "balance"]].dropna()

X_train, X_test, y_train, y_test = model_selection.train_test_split(coffee.iloc[:,1:],coffee['total_cup_points'],test_size=0.2)

lr = LinearRegression().fit(X_train, y_train)

dump(lr, 'coffeelr.joblib')