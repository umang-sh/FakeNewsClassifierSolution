# Fake News Classifier Solution
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Problem
Internet is filled with Fake news. The problem is a NLP  problem to identify fake news given any news article. The dataset is present here: 
https://www.kaggle.com/c/fake-news

The repository provides the code for following:
1.  Data Pre-Processing and cleaning
2. Exploratory Data Analysis for the Fake News Dataset
3. Multiple Feature Extraction approaches
4. Multiple Modeling Approaches, Both ML and DL.
5. Hyperparameter tuning for every approach
6. Confusion Matrix for every approach
7. Conclusion of results


## Solution Approach

### 1. Data Pre-Processing and Cleaning 
- We start by getting rid of nulls first in every feature column, we replace them with blank spaces
- Then we create a synthetic feature called "merged", which is combination of text and author columns,, this is done for 2 purposes:
    1. Classification of the news depends on both author and the content, hence a single feature would be better for learning
   2. The algorithm would only have a single feature to learn on.
   
- Moving on, we perform the following data pre-processing steps:
    1. All the punctuations/sequences are removed, they don't really help in the model learning
    2. To prevent confusion and to ease the feature extraction process, we convert everything to lower case
    3. We then make a continous stream of tokens rather than sentences
    4. Then we do stemming which is a common NLP pre-procesing step, it reduces the word to its root word for e.g chocolaty becomes chocolate   
    
### 2. Exploratory Data Analysis for the Fake News Dataset
We do few exploratory analyses via plots to understand the data better:
1. Histograms of sentence lengths
 ![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/bigrams.png)

2. 
3.





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
