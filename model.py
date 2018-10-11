import numpy as np
import tensorflow as tf


def build_deep_estimator(model_dir, hidden_units, optimizer, input_columns, run_config=None):
    # Input columns
    (visitNumber, isMobile, continent, subContinent, bounces, hits, newVisits, pageviews, visits, year, month, day,
     weekday) = input_columns

    # Turn sparse columns into one-hot
    oh_isMobile = tf.feature_column.indicator_column(isMobile)
    oh_month = tf.feature_column.indicator_column(month)
    oh_day = tf.feature_column.indicator_column(day)
    oh_weekday = tf.feature_column.indicator_column(weekday)

    # Feature cross
    month_day = tf.feature_column.crossed_column([month, day], 13 * 32)

    feature_columns = [
        # Embedding_column to "group" together
        tf.feature_column.embedding_column(continent, np.floor((6 ** 0.25))),
        tf.feature_column.embedding_column(subContinent, np.floor((23 ** 0.25))),
        tf.feature_column.embedding_column(month_day, np.floor((13 * 32) ** 0.25)),

        # One-hot encoded columns
        oh_isMobile, oh_month,
        oh_weekday, oh_day,

        # Numeric columns
        visitNumber, bounces, hits, newVisits,
        pageviews, visits, year
    ]

    estimator = tf.estimator.DNNRegressor(
        model_dir=model_dir,
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        optimizer=optimizer,
        config=run_config)

    # add extra evaluation metric for hyperparameter tuning
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)

    return estimator


def build_combined_estimator(model_dir, hidden_units, optimizer, input_columns, run_config=None):
    # Input columns
    (visitNumber, isMobile, continent, subContinent, bounces, hits, newVisits, pageviews, visits, year, month, day,
     weekday) = input_columns

    deep_columns = [
        # Numeric columns
        visitNumber, bounces, hits, newVisits,
        pageviews, visits
    ]

    # Wide columns and deep columns
    wide_columns = [
        # Sparse columns
        isMobile
    ]

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        dnn_optimizer=optimizer,
        config=run_config)

    # add extra evaluation metric for hyperparameter tuning
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)

    return estimator


def add_eval_metrics(labels, predictions):
    pred_values = predictions['predictions']
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values),
        'mae': tf.metrics.mean_absolute_error(labels, pred_values)
    }
