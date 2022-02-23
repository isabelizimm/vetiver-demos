import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from vetiver import VetiverModel, VetiverAPI, vetiver_endpoint

# Load training data
raw = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv')
df = pd.DataFrame(raw)
coffee = df[["total_cup_points", "aroma", "flavor", "sweetness", "acidity", \
    "body", "uniformity", "balance"]].dropna()

X_train, X_test, y_train, y_test = model_selection.train_test_split(coffee.iloc[:,1:],coffee['total_cup_points'],test_size=0.2)

lr = LinearRegression().fit(X_train, y_train)

coffee = VetiverModel(lr, save_ptype = True, ptype_data=X_train, model_name = "coffee")

myapp = VetiverAPI(coffee, check_ptype = True)

## add new endpoint
################
def sum_values(x):
    return x.sum()

myapp.vetiver_post(sum_values, "my_endpoint")

myapp.run()

# predict from inside script
############################
# data = {"aroma":0,"flavor":0,"sweetness":0,"acidity":0,"body":0,"uniformity":0,"balance":0}
# endpoint = vetiver_endpoint()

# response = myapp.predict(data, endpoint)
