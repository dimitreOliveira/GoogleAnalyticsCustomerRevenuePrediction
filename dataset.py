from unittest.mock import inplace

import pandas as pd
import tensorflow as tf
from pandas.io.json import json_normalize


def parse_data(csv_path='data/train_raw.csv', nrows=None):
    json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'}, nrows=nrows)

    for column in json_columns:
        df = df.join(pd.DataFrame(df.pop(column).apply(pd.io.json.loads).values.tolist(), index=df.index))

    return df


def add_engineered(features):
    # Feature engineering as data is fed
    # Nothing yet!
    return features


def read_dataset(filename, mode, features_cols, label_col, default_value, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=default_value)
            features = dict(zip(features_cols, columns))
            label = features.pop(label_col)
            return add_engineered(features), label

        # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
        filenames_dataset = tf.data.Dataset.list_files(filename)
        # Read lines from text files
        # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
        textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
        # Parse text lines as comma-separated values (CSV)
        # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)
        dataset = textlines_dataset.map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # loop indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        elif mode == tf.estimator.ModeKeys.EVAL:
            num_epochs = 1  # end-of-input after this
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()

        return batch_features, batch_labels

    return _input_fn


def pandas_train_input_fn(df, label):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=label,
        batch_size=128,
        num_epochs=100,
        shuffle=True,
        queue_capacity=1000
    )


def pandas_test_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000
    )


def output_submission(df, prediction_df, id_column, prediction_column, file_name):
    df[prediction_column] = prediction_df['predictions'].apply(lambda x: x[0])
    df[prediction_column].clip(lower=0, inplace=True)
    df = df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    df.columns = ["fullVisitorId", "PredictedLogRevenue"]
    df[[id_column, prediction_column]].to_csv(('submissions/%s' % file_name), index=False)
    print('Output complete')
