![](https://storage.googleapis.com/kaggle-media/competitions/RStudio/google_store.jpg)

## Google Analytics Customer Revenue Prediction

## About the repository
The goal of the repository is to use the competitions dataset and model data that has a large gap between min and max label values and also has lots of 0 values as labels, this should require additional effort. Also this repository is an end-to-end project using Tensorflow.

### What you will find
* Loading the json files and parsing csv format. [[link]](https://github.com/dimitreOliveira/GoogleAnalyticsCustomerRevenuePrediction/blob/master/dataset_generation.py)
* Data batch loading using Tensorflow. [[link]](https://github.com/dimitreOliveira/GoogleAnalyticsCustomerRevenuePrediction/blob/master/dataset_generation.py)
* Preprocess each batch "on the fly" with Tensorflow. [[link]](https://github.com/dimitreOliveira/GoogleAnalyticsCustomerRevenuePrediction/blob/master/dataset.py)
* Deep learning models using the estimator API from Tensorflow. [[link]](https://github.com/dimitreOliveira/GoogleAnalyticsCustomerRevenuePrediction/blob/master/model.py)
* Model train, validation and analysis using Tensorboard. [[link]](https://github.com/dimitreOliveira/GoogleAnalyticsCustomerRevenuePrediction/blob/master/tensorflow_model.py)
* Model prediction by batch with Tensorflow. [[link]](https://github.com/dimitreOliveira/GoogleAnalyticsCustomerRevenuePrediction/blob/master/tensorflow_model.py)

### Predict how much GStore customers will spend

Kaggle competition: https://www.kaggle.com/c/google-analytics-customer-revenue-prediction

### Overview
The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.

RStudio, the developer of free and open tools for R and enterprise-ready products for teams to scale and share work, has partnered with Google Cloud and Kaggle to demonstrate the business impact that thorough data analysis can have.

In this competition, you’re challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer. Hopefully, the outcome will be more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.

### Dependencies:
* [numpy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [tensorflow](https://www.tensorflow.org/)
* [pylightgbm](https://github.com/ArdalanM/pyLightGBM)

### To-Do:
* After the data leakage the competition database changed, so this work needs to be updated.
* At the time I wasn't able to properly test the LGB model.
