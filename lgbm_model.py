import datetime
import numpy as np
import pandas as pd
import pylightgbm as lgb
from dataset import add_time_features

# Unwanted columns
unwanted_columns = ['fullVisitorId', 'sessionId', 'visitId', 'visitStartTime', 'browser', 'browserSize',
                    'browserVersion', 'flashVersion', 'mobileDeviceInfo', 'mobileDeviceMarketingName',
                    'mobileDeviceModel', 'mobileInputSelector', 'operatingSystemVersion', 'screenColors', 'metro',
                    'networkDomain', 'networkLocation', 'adContent', 'campaign', 'isTrueDirect', 'keyword',
                    'referralPath', 'source', 'operatingSystem', 'day']
categorical_features = ['deviceCategory', 'isMobile', 'continent', 'month', 'weekday']
reduce_features = ['city', 'year', 'medium', 'channelGrouping', 'region', 'subContinent', 'country', 'date']
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 30,
    "min_child_samples": 100,
    "learning_rate": 0.005,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.5,
    "bagging_frequency": 5,
    "bagging_seed": 2018,
    "verbosity": -1
}

# load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Pre process pipeline
train = add_time_features(train)
test = add_time_features(test)
# Convert target feature to 'float' type.
train["transactionRevenue"] = train["transactionRevenue"].astype('float')

# Drop stange 'dict' column
train = train.drop(['adwordsClickInfo'], axis=1)
test = test.drop(['adwordsClickInfo'], axis=1)
# Drop column that exists only in train data
train = train.drop(['campaignCode'], axis=1)
# Input missing transactionRevenue values
train["transactionRevenue"].fillna(0, inplace=True)
test_ids = test["fullVisitorId"].values

train = train.drop(unwanted_columns, axis=1)
test = test.drop(unwanted_columns, axis=1)
# Constant columns
constant_columns = [c for c in train.columns if train[c].nunique() <= 1]
print('Columns with constant values: ', constant_columns)
train = train.drop(constant_columns, axis=1)
test = test.drop(constant_columns, axis=1)
# Columns with more than 50% null data
high_null_columns = [c for c in train.columns if train[c].count() <= len(train) * 0.5]
print('Columns more than 50% null values: ', high_null_columns)
train = train.drop(high_null_columns, axis=1)
test = test.drop(high_null_columns, axis=1)

train = pd.get_dummies(train, columns=categorical_features)
test = pd.get_dummies(test, columns=categorical_features)

# align both data sets (by outer join), to make they have the same amount of features,
# this is required because of the mismatched categorical values in train and test sets
train, test = train.align(test, join='outer', axis=1)

# replace the nan values added by align for 0
train.replace(to_replace=np.nan, value=0, inplace=True)
test.replace(to_replace=np.nan, value=0, inplace=True)

X_train = train[train['date'] <= datetime.date(2017, 5, 31)]
X_val = train[train['date'] > datetime.date(2017, 5, 31)]

# Get labels
Y_train = X_train['transactionRevenue'].values
Y_val = X_val['transactionRevenue'].values
X_train = X_train.drop(['transactionRevenue'], axis=1)
X_val = X_val.drop(['transactionRevenue'], axis=1)
test = test.drop(['transactionRevenue'], axis=1)
# Log transform the labels
Y_train = np.log1p(Y_train)
Y_val = np.log1p(Y_val)

X_train = X_train.drop(reduce_features, axis=1)
X_val = X_val.drop(reduce_features, axis=1)
test = test.drop(reduce_features, axis=1)

# Convert data types
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
test = test.astype('float32')

# Model
lgb_train = lgb.Dataset(X_train, label=Y_train)
lgb_val = lgb.Dataset(X_val, label=Y_val)
model = lgb.train(params, lgb_train, 10000, valid_sets=[lgb_val], early_stopping_rounds=100, verbose_eval=100)

# Predictions
predictions = model.predict(test, num_iteration=model.best_iteration)
submission = pd.DataFrame({"fullVisitorId": test_ids})
predictions[predictions < 0] = 0
submission["PredictedLogRevenue"] = predictions
submission = submission.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
submission.columns = ["fullVisitorId", "PredictedLogRevenue"]
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"]
submission.to_csv("submissions/submission.csv", index=False)
