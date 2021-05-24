# US Airlines Tweets Sentiment Analysis

Classifying a tweet as positive, neutral, or negative sentiment using Natural Language Processing (CBOW approaches) and Traditional Machine Learning Algorithms.

Highlights - 
- Reg-Ex functions have been used to pre-process the tweets.
- To convert tokens to text, we have used only sparse vectors (bag of words, counts, tf-idf, and variations). 
- Dense embeddings (e.g. word2vec, glove etc.) have not been used in this use case. (Follow the other reposiory for Dense Embeddings.
- The following Machine Learning model have been used for classification:
<u>Logistic Regression (one vs all), Naïve Bayes and XGBoost</u>

The evaluation metric for this use case is <b>Macro F1-Score</b>.

# <b>About the Data </b>

The dataset consists of Tweets and corresponding sentiment negative, neutral, or positive. 
The tweets are in the text column of the data and sentiment is in the Target column. 

The Target column has three values: 1,-1, 0 that corresponds to positive, negative, and neutral sentiment respectively. 

In this use case, our task is to train the model to predict the sentiment of the tweets.

# Experimentation & Results

| __Model__ | __Accuracy__ | __F1-Score__ |
| --------- | ------------ | ------------ |
| XGB-TFIDF-Unigram |	0.788707 |	0.725277 |
| NBC-TFIDF-Unigram |	0.646630 |	0.366959 |
| LRC-TFIDF-Unigram |	0.797814 |	0.720658 |
| XGB-TFIDF-Bigram |	0.658470 |	0.468248 |
| NBC-TFIDF-Bigram |	0.655738 |	0.404313 |
| LRC-TFIDF-Bigram |	0.660291 |	0.424098 |
| XGB-Count-Unigram |	0.770492 |	0.690770 |
| NBC-Count-Unigram |	0.750455 |	0.635276 |
| LRC-Count-Unigram |	0.794171 |	0.724309 |
| XGB-Count-BiGram |	0.663934 |	0.454908 |
| NBC-Count-BiGram |	0.686703 |	0.531020 |
| LRC-Count-BiGram |	0.679417 |	0.485213 |

# Best Model & it's parameters (Grid Search)

Total Execution Time -  606.8556914329529

Best Training Macro F1-Score -  0.7978142076502732

Best Training Params -  {'vec__max_df': 0.8, 'vec__min_df': 1, 'vec__tokenizer': <function sentiment_analysis.lemmatize at 0x0000019559293B88>}

Testing Macro F1-Score -  0.7507698534812018

Testing Accuracy -  0.8060109289617486

# Technologies & Methods
- Python 3
- Jupyter Notebooks

# Conclusion
- In order to predict the Sentiment of the Tweets, we used Traditional Machine Learning Algorithms. We started with cleaning the tweets using regular expressions. Then we tokenized the data, used Stemming and Lemmatization along with Count Vectorizer, and TF-IDF Vectorizer. 
- Once the training data was represented using the above techniques, we used Multinomial Naive Bayes, XGBoost, and Logistic Regression One Vs All. 
- After experimentation, the best results was obtained with CountVectorizer (Unigrams). Grid Search was used to search for the best hyper-parameters. 
- Then we used Cross Validation to evaluate the performance of the best classifier across the entire dataset. The average F1-Macro CV Score was 0.74 which was very close to the testing F1-Score of the best model. This helped us ensure that our model would score closely to this value across various real time datasets.

