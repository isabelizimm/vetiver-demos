import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, pipeline
from sklearn.ensemble import RandomForestRegressor

from vetiver import VetiverModel, VetiverAPI

np.random.seed(500)

raw = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-02/youtube.csv')
df = pd.DataFrame(raw)
df = df[["like_count", "brand", "year", "funny", "patriotic", \
    "celebrity", "danger", "animals", "view_count"]].dropna()
X, y = df.iloc[:,1:],df['like_count']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,test_size=0.2)

le = preprocessing.OrdinalEncoder().fit(X)
rf = RandomForestRegressor().fit(le.transform(X_train), y_train)

pipe = pipeline.Pipeline([('label_encoder',le), ('random_forest', rf)])
ads = VetiverModel(pipe, save_ptype = True, ptype_data=X_train, model_name = "superbowl_ads")

app = VetiverAPI(ads, check_ptype=True)

app.run()