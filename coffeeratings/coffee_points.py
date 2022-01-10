import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import numpy as np
from pydantic import BaseModel
import vetiver

np.random.seed(500)


raw = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv')

class Coffee(BaseModel):
    aroma: float
    flavor: float
    sweetness: float
    acidity: float
    body: float
    uniformity: float
    balance: int

df = pd.DataFrame(raw)
coffee = df[["total_cup_points", "aroma", "flavor", "sweetness", "acidity", \
    "body", "uniformity", "balance"]].dropna()

X_train, X_test, y_train, y_test = model_selection.train_test_split(coffee.iloc[:,1:],coffee['total_cup_points'],test_size=0.2)

lr = LinearRegression().fit(X_train, y_train)

vetiver.vetiver_serve(lr, Coffee)