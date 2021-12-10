# Bitcoin price prediction using sentiment analysis using transformers roberta and embeddings

## Introduction
Sentiment Analysis can be defined as the process of analyzing text data and categorizing them into Positive, Negative, or Neutral sentiments. Emotions such as anger, disgust, fear, happiness, sadness, and surprise can also be detected. Sentiment Analysis is used in many cases like tracking social media conversations, product reviews, feedbacks, survey responses, etc. which allows companies to understand the customer's emotions better and thus meet their needs.

## About the problem statement
Bitcoin has become a very popular investing instrument in recent times with the promotion of decentralized digital currency without a central bank or single administrator. This leads to price fluctuations, which can also be attributed to positive or negative twitter conversations around bitcoin. Here, I attempt to predict bitcoin price movement by assessing twitter sentiment.

## Methodology
Bitcoin tweets data is used to predict bitcoin prices. 2 files containing bitcoin tweets data have around 8,00,000 and 10,00,000 rows each. Due to the sheer volume of data, deep learning approach isused. Every second we have around 40 tweets on bitcoin in the data. Bitcoin price might not fluctuate very much every second with any useful predictive ability, so I combine all tweets into per minute and check for price fluctuations every minute.
# 
Method 1 (Using sentiment score) - Due to compute restrictions while running transformer model for sentiment extraction, I predict one day's bitcoin price fluctuations for minute timeframe - which gives me a total of around 65,000 rows of tweets to analyze. For sentiment extraction from tweets, I used 'twitter-xlm-roberta-base-sentiment' for extracting sentiment score between -1 to 1 -> -1 for negative, 0 for neutral and 1 for positive sentiment. We also get the probability score associated with each sentiment score. Using this, I predict whether bitcoin price will go up or down using stacked dense layers.
#
Method 2  (Using embeddings) - In the second method, instead of sending the sentiment score to the model as input, I convert the tweet's text into embeddings using 'sentence-transformers/all-mpnet-base-v2' model. Then the model can identify hidden features from the data and predict bitcoin price. I have experimented with changing the hyperparameters in the model like number of nodes in dense layer, number of dropout layers, dropout rate, etc. to compare the accuracy.

##    Models Accuracy Comparison
######       Model with Sentiment score      --               68% Accuracy
######       Model with Embeddings - 10 nodes   --            65% Accuracy
######       Model with Embeddings - 50 nodes    --           61% Accuracy
######       Model with Embeddings - 100 nodes    --          62% Accuracy
######       Model with Embeddings - 400 nodes with 2 dropout layers -- 62% Accuracy

#
## Improvements: 
More data could improve the model --- Include information from other sources - financial news --- Consider correlation with other stock prices --- Consider other stock prices in input --- Use improved sentiment model --- Consider smaller time stamps - minite/30 mins/6 hour/12 hour/24 hour --- Take into consideration other crypto currency prices --- Consider moving averages column
