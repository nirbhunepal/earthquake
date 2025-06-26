import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


df = pd.read_csv("Earthquake.csv")
df = df.dropna()
X = df[['latitude', 'longitude', 'depth']]
y = df['mag']
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, 'model.joblib')