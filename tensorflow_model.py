from model import *
from dataset import *

tf.logging.set_verbosity(tf.logging.INFO)

# Parameters
TRAIN_PATH = 'data/tf_train.csv'
VALIDATION_PATH = 'data/tf_validation.csv'
TEST_PATH = 'data/test.csv'
MODEL_DIR = 'models/model1'
SUBMISSION_NAME = 'submission1.csv'

LEARNING_RATE = 0.0001
HIDDEN_UNITS = [32, 32, 16]
STEPS = 30000
BATCH_SIZE = 512
CSV_COLUMNS = ['fullVisitorId', 'visitNumber', 'bounces', 'hits', 'newVisits', 'pageviews', 'transactionRevenue',
               'visits', 'year',
               'month', 'day', 'weekday']
LABEL_COLUMN = 'transactionRevenue'
DEFAULTS = [['default_id'], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [1.0], [2016.0], [9.0], [2.0], [4.0]]
INPUT_COLUMNS = [
    # raw data columns
    tf.feature_column.numeric_column('visitNumber'),
    tf.feature_column.numeric_column('bounces'),
    tf.feature_column.numeric_column('hits'),
    tf.feature_column.numeric_column('newVisits'),
    tf.feature_column.numeric_column('pageviews'),
    tf.feature_column.numeric_column('visits')
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
             'bounces': 'float32',
             'hits': 'float32',
             'newVisits': 'float32',
             'pageviews': 'float32',
             'visits': 'float32',
             'year': 'float32',
             'month': 'float32',
             'day': 'float32',
             'weekday': 'float32'}

test_raw = pd.read_csv(TEST_PATH, dtype=datatypes)
prediction = estimator.predict(pandas_test_input_fn(test_raw))
prediction_df = pd.DataFrame(prediction)
output_submission(test_raw, prediction_df, 'fullVisitorId', 'PredictedLogRevenue', SUBMISSION_NAME)