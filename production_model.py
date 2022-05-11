import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train_num=train.select_dtypes(['int64','float64'])
test_num=test.select_dtypes(['int64','float64'])


y = train_num.revenue        
train_num.drop(['revenue'], axis=1, inplace=True)

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer(strategy='mean')),
    ('model', RandomForestRegressor(n_estimators=200, random_state=0))
])
print("Fitting model to training data")
my_pipeline.fit(train_num,y)

preds=my_pipeline.predict(test_num)

joblib.dump(my_pipeline, 'model.pkl')
print("Dumped model to model.pkl file")
features = list(train_num.columns)
joblib.dump(features, 'model_columns.pkl')
print("Dumped features to model_columns.pkl file")