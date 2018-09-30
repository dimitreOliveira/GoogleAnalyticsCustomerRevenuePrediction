from dataset import parse_data, add_time_features


# Script to load, parse and process data.
unwanted_columns = ['channelGrouping', 'fullVisitorId', 'sessionId', 'socialEngagementType', 'visitId',
                    'visitStartTime', 'browser', 'browserSize', 'browserVersion', 'deviceCategory', 'flashVersion',
                    'language', 'mobileDeviceBranding', 'mobileDeviceInfo', 'mobileDeviceMarketingName',
                    'mobileDeviceModel', 'mobileInputSelector', 'operatingSystem', 'operatingSystemVersion',
                    'screenColors', 'screenResolution', 'city', 'cityId', 'metro', 'networkDomain', 'networkLocation',
                    'adContent', 'adwordsClickInfo', 'campaign', 'isTrueDirect', 'keyword', 'medium',
                    'referralPath', 'source', 'latitude', 'longitude']

train = parse_data('data/train_raw.csv')
train.drop(['campaignCode'], axis=1, inplace=True)
train.drop(unwanted_columns, axis=1, inplace=True)
train = add_time_features(train)
train.drop(['date'], axis=1, inplace=True)
train.to_csv('data/train.csv', index=False)

test = parse_data('data/test_raw.csv')
test.drop(unwanted_columns, axis=1, inplace=True)
test = add_time_features(test)
test.drop(['date'], axis=1, inplace=True)
test.to_csv('data/test.csv', index=False)
