from dataset import parse_data, add_time_features


# Script to load, parse and process data.

unwanted_columns = ['adwordsClickInfo', 'fullVisitorId', 'sessionId', 'visitId', 'visitStartTime', 'browser',
                    'browserSize', 'browserVersion', 'flashVersion', 'mobileDeviceInfo', 'mobileDeviceMarketingName',
                    'mobileDeviceModel', 'mobileInputSelector', 'operatingSystemVersion', 'screenColors', 'metro',
                    'networkDomain', 'networkLocation', 'adContent', 'campaign', 'isTrueDirect', 'keyword',
                    'referralPath', 'source', 'operatingSystem', 'medium', 'channelGrouping', 'date']

train = parse_data('data/train_raw.csv')
train = add_time_features(train)
train = train.drop(unwanted_columns, axis=1)
train.to_csv('data/train.csv', index=False)

test = parse_data('data/test_raw.csv')
test = add_time_features(test)
test = test.drop(unwanted_columns, axis=1)
test.to_csv('data/test.csv', index=False)