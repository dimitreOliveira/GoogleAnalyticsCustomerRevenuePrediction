import csv
import numpy as np
import pandas as pd
from dataset import parse_data


def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df = df.drop(['date'], axis=1)

    return df


def split_data(input_data_path='data/train.csv', train_data_path='data/tf_train.csv',
               validation_data_path='data/tf_validation.csv', ratio=10):
    """
    Splits the csv file (meant to generate train and validation sets).
    :param input_data_path: path containing the full data set.
    :param train_data_path: path to write the train set.
    :param validation_data_path: path to write the validation set.
    :param ratio: ration to split train and validation sets, (default: 1 of every 10 rows will be validation or 10%)
    """
    with open(input_data_path, 'r') as inp, open(train_data_path, 'w', newline='') as out1, \
            open(validation_data_path, 'w', newline='') as out2:
        csv_reader = csv.reader(inp)
        # Skip header
        next(csv_reader)
        writer1 = csv.writer(out1)
        writer2 = csv.writer(out2)
        count = 0
        for row in csv_reader:
            if count % ratio == 0:
                writer2.writerow(row)
            else:
                writer1.writerow(row)
            count += 1


# Script to load, parse and process data.
unwanted_columns = ['channelGrouping', 'sessionId', 'socialEngagementType', 'visitId',
                    'visitStartTime', 'browser', 'browserSize', 'browserVersion', 'deviceCategory', 'flashVersion',
                    'language', 'mobileDeviceBranding', 'mobileDeviceInfo', 'mobileDeviceMarketingName',
                    'mobileDeviceModel', 'mobileInputSelector', 'operatingSystem', 'operatingSystemVersion',
                    'screenColors', 'screenResolution', 'city', 'cityId', 'metro', 'networkDomain', 'networkLocation',
                    'adContent', 'adwordsClickInfo', 'campaign', 'isTrueDirect', 'keyword', 'medium',
                    'referralPath', 'source', 'latitude', 'longitude', 'continent', 'country', 'region', 'subContinent']

train = parse_data('data/train_raw.csv')
train.to_csv('data/train_parsed.csv', index=False)
train.drop(['campaignCode'], axis=1, inplace=True)
train.drop(unwanted_columns, axis=1, inplace=True)
train = add_time_features(train)
train["transactionRevenue"] = train["transactionRevenue"].astype('float')
train["transactionRevenue"].fillna(0, inplace=True)
train.dropna(subset=['transactionRevenue'], inplace=True)
train["transactionRevenue"] = np.log1p(train["transactionRevenue"])
train.to_csv('data/train.csv', index=False)

test = parse_data('data/test_raw.csv')
test.to_csv('data/test_parsed.csv', index=False)
test.drop(unwanted_columns, axis=1, inplace=True)
test = add_time_features(test)
test.to_csv('data/test.csv', index=False)

# Generate train and validation files
split_data()
