# Fake News Classifier Solution
## _This repository contains the code and results for Fake News classifier problem_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The problem is a text classification problem and the solution Jupyter Notebook is broken down into 3 steps

- Feature Engineering
- Data Pre-Processing
- Model Building and Evaluation

## Solution Approach

- We start by getting rid of nulls first in every feature column, we replace them with blank spaces
- Then we create a synthetic feature called "merged", which is combination of text and author columns, the reason of this is also written in the Jupyter notebook markdown
- Moving on, we perform data pre-processing steps, such as punctutation removal, stemming etc. The details can be found in the notebook
- We then perform padding to make sure every input feature vector is of the same length
- Lastly, we train and look at the metrics of 2 models
- We first try a ML model , A tree based model with Boosting called XG boost, we then try a Deep learning model as well, since LSTMs have proven to work very well in text classification scenarios.




| Model | Precision | Recall
| ------ | ------ | ------ |
| XGBoost (0-class) |0.99   | 0.98 |
| XGBoost (1-class)| 0.98 |0.99 |
| Deep learning model (0-class) |0.98   | 0.99 |
| Deep learning model(1-class)| 0.99 |0.98 |


| Model | Accuracy 
| ------ | ------|
| XGBoost  |0.98   |
| Deep Learning Model  |0.99   |


## The repository contains following files:
1. FakeNewsClassifier.ipynb 
2. Submission.csv
3. README.md

#### The jupyter notebook assumes the data is downloaded and extracted in the following path data/fake-news/
