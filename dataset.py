import pandas as pd
from pandas.io.json import json_normalize


def parse_data(csv_path='data/train_raw.csv', nrows=None):
    json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'}, nrows=nrows)

    for column in json_columns:
        df = df.join(pd.DataFrame(df.pop(column).apply(pd.io.json.loads).values.tolist(), index=df.index))

    return df


def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())

    return df


# # Script
# train = parse_data('data/train_raw.csv')
# train.to_csv('data/train.csv', index=False)
# test = parse_data('data/test_raw.csv')
# test.to_csv('data/test.csv', index=False)
