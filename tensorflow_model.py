from model import *
from dataset import *

tf.logging.set_verbosity(tf.logging.INFO)

# Parameters
TRAIN_PATH = 'data/tf_train.csv'
VALIDATION_PATH = 'data/tf_validation.csv'
TEST_PATH = 'data/test.csv'
MODEL_NAME = 'model17'
MODEL_DIR = 'models/' + MODEL_NAME
SUBMISSION_NAME = ('submission_%s.csv' % MODEL_NAME)

# Model parameters
LEARNING_RATE = 0.0001
HIDDEN_UNITS = [128, 64, 32, 16]
STEPS = 100000
BATCH_SIZE = 512


VOCAB_MOBILE = ['True', 'False']
VOCAB_CONTINENT = ['Asia', 'Oceania', 'Europe', 'Americas', 'Africa', '(not set)']
CSV_COLUMNS = ['fullVisitorId', 'visitNumber', 'isMobile', 'continent', 'subContinent', 'bounces', 'hits', 'newVisits',
               'pageviews', 'transactionRevenue', 'visits', 'year', 'month', 'day', 'weekday']
LABEL_COLUMN = 'transactionRevenue'
DEFAULTS = [['default_id'], [0.0], ['False'], ['continent_not_set'], ['subContinent_not_set'], [0.0], [0.0], [0.0],
            [0.0], [0.0], [0.0], [2016], [1], [1], [1]]
INPUT_COLUMNS = [
    # Raw data columns
    tf.feature_column.numeric_column('visitNumber'),
    tf.feature_column.categorical_column_with_vocabulary_list('isMobile', vocabulary_list=VOCAB_MOBILE),
    tf.feature_column.categorical_column_with_vocabulary_list('continent', vocabulary_list=VOCAB_CONTINENT),
    tf.feature_column.categorical_column_with_hash_bucket('subContinent', hash_bucket_size=30),
    tf.feature_column.numeric_column('bounces'),
    tf.feature_column.numeric_column('hits'),
    tf.feature_column.numeric_column('newVisits'),
    tf.feature_column.numeric_column('pageviews'),
    tf.feature_column.numeric_column('visits'),
    tf.feature_column.numeric_column('year'),
    tf.feature_column.categorical_column_with_identity('month', num_buckets=13),
    tf.feature_column.categorical_column_with_identity('day', num_buckets=32),
    tf.feature_column.categorical_column_with_identity('weekday', num_buckets=7)
]

train_spec = tf.estimator.TrainSpec(input_fn=read_dataset(TRAIN_PATH, mode=tf.estimator.ModeKeys.TRAIN,
                                                          features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
                                                          default_value=DEFAULTS, batch_size=BATCH_SIZE),
                                    max_steps=STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn=read_dataset(VALIDATION_PATH, mode=tf.estimator.ModeKeys.EVAL,
                                                        features_cols=CSV_COLUMNS, label_col=LABEL_COLUMN,
                                                        default_value=DEFAULTS, batch_size=BATCH_SIZE),
                                  steps=1000, throttle_secs=300)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
estimator = build_deep_estimator(MODEL_DIR, HIDDEN_UNITS, optimizer, INPUT_COLUMNS)

tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)

# Make predictions
datatypes = {'fullVisitorId': 'str',
             'visitNumber': 'float32',
             'isMobile': 'str',
             'continent': 'str',
             'subContinent': 'str',
             'bounces': 'float32',
             'hits': 'float32',
             'newVisits': 'float32',
             'pageviews': 'float32',
             'visits': 'float32',
             'year': 'int32',
             'month': 'int32',
             'day': 'int32',
             'weekday': 'int32'}

test_raw = pd.read_csv(TEST_PATH, dtype=datatypes)
prediction = estimator.predict(pandas_test_input_fn(test_raw))
prediction_df = pd.DataFrame(prediction)
output_submission(test_raw, prediction_df, 'fullVisitorId', 'PredictedLogRevenue', SUBMISSION_NAME)
