# Fake News Classifier Solution
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Problem
Internet is filled with Fake news. The problem is a NLP  problem to identify fake news given any news article. Its a binary text classification problem. The dataset is present here: 
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

1. Histograms of sentence lengths. This tells us the lengths of both reliable and unreliable news articles and it can be clearly seen that reliable articles are shorter in length.

 ![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/histogramoflengths.png)

2. Secondly,we look at the distribution of labels, i.e how many reliable vs unreliable news articles do we have in the data. Data has almost equal amount of both reliable and unreliable news, this tells us we won't have a **Class Imbalance problem**

![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/classdistribution.png)

3. We then look at the counts of bigrams and trigrams in the text. It gives us following insights:
   - New York is the most common bigram in the reliable news and hillari clinton is most common one in unreliable
   - New york time is the most common trigam in the reliable and norepli blogger com in the unreliable one
Shown below is one such bigram plot:

![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/bigrams.png)

4. Lastly we look at word clouds,we look at the word cloud of both reliable and unreliable news, word clouds give us a pretty good picture of the terms present in the data. The size of the words in the word cloud is proportional to how frequently they appear in the data
Shown below is the word cloud for reliable news articles:

![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/reliablewordcloud.png)


### 3. Multiple Feature Extraction approaches.
We try 4 NLP Feature extraction approaches individually

#### 1. Count Vectorizer:
Machines cannot understand characters and words. So when dealing with text data we need to represent it in numbers to be understood by the machine. Countvectorizer is a method to convert text to numerical data. An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document.

We use CountVectorizer as the first approach and extracted features are passed to MultinomialNB model.

#### 2. TF-IDF:
TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval,

We use tf-idf as the second approach and extracted features are passed to MultinomialNB model.

#### 3. Hashing:
Counts and frequencies can be very useful, but one limitation of these methods is that the vocabulary can become very large.
This, in turn, will require large vectors for encoding documents and impose large requirements on memory and slow down algorithms.
A clever work around is to use a one way hash of words to convert them to integers.

We use hashing as the third approach and extracted features are passed to MultinomialNB model.

#### 4. Word2Vec:
Now we try a Deep learning based Feature extraction approach as well as the Deep learning Model.
Word embedding is one of the document representation in vector space model. It captures contexts and semantics of word unlike Bag-of-Words model. 
Word2vec is one of the most popular implementation of word embedding, which is invented by Google in 2013. It describes word embedding with two-layer shallow neural networks in order to recognize context meanings.
There are 2 ways to create word embeddings:

1. Using word embedding in an unsupervised way: This means that, based on the sentences we have, the model tries to capture the context of a word in a document, that is, the relation to other words, semantic and syntactic similarities, etc...
2. Using word embedding as a first layer in the neural network/LSTM. This means that the weights are learned through backpropagation of the classification error. Therefore, weights are learned by backpropagating the classification error. The model does not necessarily try to capture the context of a word in a document, it adjusts the weights only in such a way as to reduce loss.

We will use here the first method. It is faster because the embedding only needs to be done once, and not on each iteration of the LSTM training. An advantage of the first method is the possibility of using pre-trained models, that is, models which have already associated the words with vectors.

We could have also used pre-trained embeddings instead of training our own using gensims' word2vec using the stored model here  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing 
 but they need a good amount of memory(~12GB+ RAM) and space to load which we dont have.

### 4. Modelling, Hyperparameter Tuning and Results.
Majorly we try 2 Models in different combinations with Feature extraction approaches defined above:

#### 1. Multinomial NB:
In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features.
Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.

#### 2. LSTM:
LSTM or long short term memory is a type of RNN that was developed to solve the problem of vanishing gradients, exploding gradients and is very good in understanding long term and short term dependencies among the text data. It has 3 gates that control these operations.


We try 4 combinations of feature extraction and models and also do hyperparameter search for each of them except Deep learning model due to compute limitations. Details of implementation of each approach can be found in the Jupyter Notebook.
We try the following combinations:
| Feature Extraction | Model |
| :- | -: | 
| Count Vectorizer | Multinomial Naive Bayes 
| TF-IDF| Multinomial Naive Bayes 
| Hashing | Multinomial Naive Bayes 
| Word2Vec based Embedding | LSTM based Neural Net 

Lets compare the results of each of them and also see confusion matrix for each:

#### 1. CountVectorizer + MultiNomialNB
We achieve the max accuracy of 94.7 % hyperparameter tuning with this combination with below confusion matrix:
![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/confusionm1.png)

#### 2. TF-IDF + MultiNomialNB:
We achieve the max accuracy of 93 % after hyperparameter tuning with this combination with below confusion matrix:
![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/confusionm2.png)

#### 3. Hashing + MultiNomialNB:
We achieve the max accuracy of 93 % hyperparameter tuning with this combination with below confusion matrix:
![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/confusionm3.png)

#### 4. Word2Vec+ LSTM:
We create a simple Neural network with Embedding as the input and 1 LSTM unit followed by a dense layer  for results. For word embeddings, we use the Keras Embedding layer and make it untrainable since we already created that with gensim's word2vec.

We acheive the best possible accuracy **98 %**  with this approach with follwing confusion matrix:
![alt text](https://github.com/umang-sh/FakeNewsClassifierSolution/blob/main/screenshots/confusionm4.png)


### Conclusion:
In conclusion, we tried multiple feature extraction approaches, better the feature extraction usually better the results are. Also choice of model becomes important after finding a good representation of data.

Word2Vec+LSTM gives us the best results. The reason for that is that Word Embeddings captures all the semantics behind words and also the relationship between the words very well and provided a good representation of words. Also LSTM being an RNN works very well with texts and captures long term relationships and short term relationships amongst words without having problems of exploding and vanishing gradients.
