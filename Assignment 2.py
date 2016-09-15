
# coding: utf-8

# In[27]:

####### Subtask  A ######
import nltk
import csv 
import os
import random
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
#setting the directory in which the files will be found
os.chdir("/Users/Akhilesh Pandey/Desktop/art_corpus/ART_Corpus")

# reading the file as .csv using pandas. This is to ensure the data is imported as a data frame
train = pd.read_csv("twitter-train-cleansed-A.tsv", header= None, sep="\t")

# reading the test/development file, dev-input A, which will be used to tune the parameters of our classifier
test = pd.read_csv("twitter-test-A.tsv", header=None, delimiter="\t", quoting=3 )

# assigning the variable 'stop' to the set of english stopwords downloaded from the nltk package
stop=set(stopwords.words('english'))
# Removing stopwords that could be indicative of sentiment
stop.remove('t')
stop.remove('s')
stop.remove('no')
stop.remove('not')
stop.remove('nor')
stopwordList=[]

# defining a function that replaces multiple repeated characters with only two of the same characters -- e.g. "coool"
# becomes "cool"
def onlyone(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# defining the function that will pre-process the Tweets
def process_tweets(ourTweets):
    
    # Converting all the uppercase letters in the Tweets to lowercase
    ourTweets=ourTweets.lower()
    
    # Removing/stripping forward slashes, and speech marks
    ourTweets=ourTweets.strip('"/')
    
    # Removing hashtags
    ourTweets=ourTweets.replace("#","")
    
    # Removing the punctuations in the Tweets -- e.g. "3.99!!!?!" becomes "3.99"
    ourTweets=re.sub("(\?+)?\!+(\?+)?|(\?)+|(\,,+)|(\.)|(\,)|(\;)|(\w+:\/\/\S+)",'',ourTweets)
    
    # Replacing all the URLs from the Tweets with the word URL
    ourTweets=re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', ourTweets)
    
    # Removing all @usernames from our tweets as they are not informative of sentiment
    ourTweets=re.sub('@[^\s]+','',ourTweets)
    
    # Replacing all possible combinations of happy/positive emoticons with the string "hemoticons"
    
    ourTweets=re.sub('(\:[d])|(\:\))|(\(\:)|(\:(.)?\))|(\((.)?\:)',' hemoticons ', ourTweets)
    
    
    # Replacing all possible combinations of sad/negative emoticons with the string "semoticons"
    ourTweets=re.sub('(\:\()|(\)\:)|(\:(.)?\()|(\)(.)?\:)',' semoticons ',ourTweets)
    
    # Removing apostrophes from the Tweets
    ourTweets = ourTweets.replace("'", "")
    
    # Calling onlyone function to replace multiple repeated characters with a single character
    ourTweets=onlyone(ourTweets)
    
    # Changing plural words to singular by removing the extra "s" at the end
    s=".."
    for i in range(1,10):
        ourTweets=ourTweets.replace(s, " ")
        s=s+ ".+"
        
    # Splitting the string of Tweet into separate words
    words = ourTweets.split()

    # Adding words "URL" onto the stop set and removing them from the Tweets
    for i in words:
        if i=='URL':
            words.remove(i)
            stopwordList.append(i)
    
    # Removing the stopwords from the Tweets and joining the words into a string
    for i in words:
        if i in stop:
            words.remove(i)
    
    # Removing any blank spaces or extra brackets, which would be the result of previous processing
    for i in words:
        if i=='' or i==")" or i=="(":
            words.remove(i)
        
    return( " ".join( words ))

# Naming the 5th column of the training set, "tweet"
train['tweet']=train[5]

RL=[]
x=[]
clean_instances=[]
clean_tweet_instances=[]

# List 'x' holds the text of the tweet in words (not as a full string)
for i in range(len(train)):
    RL = train['tweet'][i].split()
    x.append(RL)

# Setting boundaries for the instances we want to consider for each tweet -- given by the second and third column values
# in our training set
for i in range(len(train)):
    ii = int(train[2][i])
    fi = int(train[3][i])
    str1 =' '.join(x[i][ii :fi+1])
    clean_instances.append(str1)

# Calling the process_tweets() function for every tweet and appending the processed tweets in a list called 
# clean_tweet_instances
for i in range(len(train)):
    str1=clean_instances[i]
    str2=process_tweets(str1)
    clean_tweet_instances.append(str2)


# Initialising the list to store the processed Tweets from the test data 
clean_test_instances = []
RL_test = []
x_test= []
clean_test_tweets_instances=[]

# Similarly for the test set, list 'x_test' holds the text of the tweet in words (not as a full string)
for i in range(len(test)):
    RL_test = test[5][i].split()
    x_test.append(RL_test)

# Setting boundaries for the instances we want to consider for each tweet -- given by the second and third column values
# in our testing/development set
for i in range(len(test)):
    iit = int(test[2][i])
    fit = int(test[3][i])
    ii1 = iit-1
    fi1 = fit+1
    str1 =' '.join(x_test[i][iit :fit+1])
    if ii1>=0 and fi1<=len(x_test[i])-1:
        str1 =' '.join(x_test[i][ii1 :fi1+1])
    clean_test_instances.append(str1)

# Calling the process_tweets() function for every tweet and appending the processed test tweets in a list called 
# clean_test_tweets_instances
for i in range(len(test)):
    str1=clean_test_instances[i]
    str2=process_tweets(str1)
    clean_test_tweets_instances.append(str2)

# Replacing any possible spaces within the processed tweet instances with a randomly chosen word within the list
for i in range(len(clean_test_tweets_instances)):
    if clean_test_tweets_instances[i]=='':
        clean_test_tweets_instances[i]=random.choice(clean_test_tweets_instances)


        
#### Building a feature vector using CountVectorizer ####

# Using CountVectorizer to build a features vector - specifying that the vector is made of word ngrams rather than 
# characters, removing the stopwords and initialising at max_features=5000.
vectorizer = CountVectorizer(analyzer = "word",ngram_range=(1,3),min_df=1,stop_words=stop, max_features=10000)

# Learning the vocabulary dictionary and returning term-document matrix
train_data_features = vectorizer.fit_transform(clean_tweet_instances)

# Converting the feature matrix to an array
train_data_features = train_data_features.toarray()

# Learning the vocabulary dictionary for the test set of instances and returning term-document matrix
test_data_features = vectorizer.transform(clean_test_tweets_instances)

# Converting the feature matrix to an array     
test_data_features = test_data_features.toarray()



##### Building the Random Forest classifier #####

# Initialising the Random Forest Classifier with 'n' estimators=20 to begin with
forest = RandomForestClassifier(n_estimators = 20) 

# Fitting training data to the Random Forest model
forest = forest.fit(train_data_features, train[4])

# Predicting the sentiment labels of dev-input-A, based on model built from training data, train-cleansed-A
result = forest.predict(test_data_features)

# Converting the results to data frame and combining id with labels
output = pd.DataFrame( data={"id":test[0], "sentiment":result} )


#### F1 Metric for evaluation ####

# Calculating the f1 score -- comparing mood predictions for 'dev-input-A' to the correct ones present in 'dev-gold-A'
# and tuning the parameters of the model accordingly to improve our results
####y_true= pd.read_csv("twitter-dev-gold-A.tsv", header= None, sep="\t")
####q = y_true[4]
####f1_score(q , result,average='macro')


# In[ ]:

##Testing on Test Data and prinitng the output in the Excel file
#We are using countVectorizer as it is giving the highest accuracy in Subtask A
output = pd.DataFrame( data={"id":test[1], "sentiment":result} )
writer = pd.ExcelWriter('Output_Subtask_A.xlsx', engine='xlsxwriter')
output.to_excel(writer, sheet_name='Sheet1')
writer.save()


# In[ ]:

####### Subtask  A ######
import nltk
import csv 
import os
import random
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
#setting the directory in which the files will be found
os.chdir("/Users/Akhilesh Pandey/Desktop/art_corpus/ART_Corpus")

# reading the file as .csv using pandas. This is to ensure the data is imported as a data frame
train = pd.read_csv("twitter-train-cleansed-A.tsv", header= None, sep="\t")

# reading the test/development file, dev-input A, which will be used to tune the parameters of our classifier
test = pd.read_csv("twitter-dev-input-A.tsv", header=None, delimiter="\t", quoting=3 )

# assigning the variable 'stop' to the set of english stopwords downloaded from the nltk package
stop=set(stopwords.words('english'))
# Removing stopwords that could be indicative of sentiment
stop.remove('t')
stop.remove('s')
stop.remove('no')
stop.remove('not')
stop.remove('nor')
stopwordList=[]

#### DATA PREPROCESSING ####
# defining a function that replaces multiple repeated characters with only two of the same characters -- e.g. "coool"
# becomes "cool"
def onlyone(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# defining the function that will pre-process the Tweets
def process_tweets(ourTweets):
    
    # Converting all the uppercase letters in the Tweets to lowercase
    ourTweets=ourTweets.lower()
    
    # Removing/stripping forward slashes, and speech marks
    ourTweets=ourTweets.strip('"/')
    
    # Removing hashtags
    ourTweets=ourTweets.replace("#","")
    
    # Removing the punctuations in the Tweets -- e.g. "3.99!!!?!" becomes "3.99"
    ourTweets=re.sub("(\?+)?\!+(\?+)?|(\?)+|(\,,+)|(\.)|(\,)|(\;)|(\w+:\/\/\S+)",'',ourTweets)
    
    # Replacing all the URLs from the Tweets with the word URL
    ourTweets=re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', ourTweets)
    
    # Removing all @usernames from our tweets as they are not informative of sentiment
    ourTweets=re.sub('@[^\s]+','',ourTweets)
    
    # Replacing all possible combinations of happy/positive emoticons with the string "hemoticons"
    ourTweets=re.sub('(\:[d])|(\:\))|(\(\:)|(\:(.)?\))|(\((.)?\:)',' hemoticons ', ourTweets)
    # Replacing all possible combinations of sad/negative emoticons with the string "semoticons"
    ourTweets=re.sub('(\:\()|(\)\:)|(\:(.)?\()|(\)(.)?\:)',' semoticons ',ourTweets)
    
    # Removing apostrophes from the Tweets
    ourTweets = ourTweets.replace("'", "")
    
    # Calling onlyone function to replace multiple repeated characters with a single character
    ourTweets=onlyone(ourTweets)
    
    # Changing plural words to singular by removing the extra "s" at the end
    s=".."
    for i in range(1,10):
        ourTweets=ourTweets.replace(s, " ")
        s=s+ ".+"
        
    # Splitting the string of Tweet into separate words
    words = ourTweets.split()

    # Adding words "URL" onto the stop set and removing them from the Tweets
    for i in words:
        if i=='URL':
            words.remove(i)
            stopwordList.append(i)
    
    # Removing the stopwords from the Tweets and joining the words into a string
    for i in words:
        if i in stop:
            words.remove(i)
    
    # Removing any blank spaces or extra brackets, which would be the result of previous processing
    for i in words:
        if i=='' or i==")" or i=="(":
            words.remove(i)
        
    return( " ".join( words ))

# Naming the 5th column of the training set, "tweet"
train['tweet']=train[5]

RL=[]
x=[]
clean_instances=[]
clean_tweet_instances=[]

# List 'x' holds the text of the tweet in words (not as a full string)
for i in range(len(train)):
    RL = train['tweet'][i].split()
    x.append(RL)

# Setting boundaries for the instances we want to consider for each tweet -- given by the second and third column values
# in our training set
for i in range(len(train)):
    ii = int(train[2][i])
    fi = int(train[3][i])
    str1 =' '.join(x[i][ii :fi+1])
    clean_instances.append(str1)

# Calling the process_tweets() function for every tweet and appending the processed tweets in a list called 
# clean_tweet_instances
for i in range(len(train)):
    str1=clean_instances[i]
    str2=process_tweets(str1)
    clean_tweet_instances.append(str2)


# Initialising the list to store the processed Tweets from the test data 
clean_test_instances = []
RL_test = []
x_test= []
clean_test_tweets_instances=[]

# Similarly for the test set, list 'x_test' holds the text of the tweet in words (not as a full string)
for i in range(len(test)):
    RL_test = test[5][i].split()
    x_test.append(RL_test)

# Setting boundaries for the instances we want to consider for each tweet -- given by the second and third column values
# in our testing/development set
for i in range(len(test)):
    iit = int(test[2][i])
    fit = int(test[3][i])
    ii1 = iit-1
    fi1 = fit+1
    str1 =' '.join(x_test[i][iit :fit+1])
    if ii1>=0 and fi1<=len(x_test[i])-1:
        str1 =' '.join(x_test[i][ii1 :fi1+1])
    clean_test_instances.append(str1)

# Calling the process_tweets() function for every tweet and appending the processed test tweets in a list called 
# clean_test_tweets_instances
for i in range(len(test)):
    str1=clean_test_instances[i]
    str2=process_tweets(str1)
    clean_test_tweets_instances.append(str2)

# Replacing any possible spaces within the processed tweet instances with a randomly chosen word within the list
for i in range(len(clean_test_tweets_instances)):
    if clean_test_tweets_instances[i]=='':
        clean_test_tweets_instances[i]=random.choice(clean_test_tweets_instances)


        
#### Tfidf feature vector ####
# Using TfidfVectorizer to build a features vector - specifying that the vector is made of word ngrams rather than 
# characters, removing the stopwords and initialising at max_features=5000.
tf_vectorizer = TfidfVectorizer(analyzer = "word", ngram_range=(1,3), min_df=1, stop_words=stop, max_features=3000)

# Learning the vocabulary dictionary and returning term-document matrix
train_data_features = tf_vectorizer.fit_transform(clean_tweet_instances)

# Converting the feature matrix to an array
train_data_features = train_data_features.toarray()

# Learning the vocabulary dictionary for the test set of instances and returning term-document matrix
test_data_features = tf_vectorizer.transform(clean_test_tweets_instances)

# Converting the feature matrix to an array       
test_data_features = test_data_features.toarray()



##### Building the Random Forest classifier #####

# Initialising the Random Forest Classifier with 'n' estimators=20 to begin with
forest = RandomForestClassifier(n_estimators = 1000) 

# Fitting training data to the Random Forest model
forest = forest.fit(train_data_features, train[4])

# Predicting the sentiment labels of dev-input-A, based on model built from training data, train-cleansed-A
result = forest.predict(test_data_features)

# Converting the results to data frame and combining id with labels
output = pd.DataFrame( data={"id":test[0], "sentiment":result} )



#### F1 Metric for evaluation ####

# Calculating the f1 score -- comparing mood predictions for 'dev-input-A' to the correct ones present in 'dev-gold-A'
# and tuning the parameters of the model accordingly to improve our results
y_true= pd.read_csv("twitter-dev-gold-A.tsv", header= None, sep="\t")
q = y_true[4]
f1_score(q , result,average='macro')


# In[ ]:

############      SUBTASK B     ############

import nltk
import csv 
import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score

# Sets/changes the directory to the document directory
os.chdir("/Users/Akhilesh Pandey/Desktop/art_corpus/ART_Corpus")
# Reads the training .tsv file and saves it as train
train = pd.read_csv("twitter-train-cleansed-B.tsv", header= None, sep="\t")
# Reads the testing/tuning .tsv file and saves it as test
test = pd.read_csv("twitter-dev-input-B.tsv", header=None, delimiter="\t", quoting=3 )

# Initialising the list to store the processed Tweets from the trainig data 
clean_train_tweets = []
# Initialising the list to store the processed Tweets from the test data
clean_test_tweets = []


# assigning the variable 'stop' to the set of english stopwords downloaded from the nltk package
stop=set(stopwords.words('english'))
# Removing stopwords that could be indicative of sentiment
stop.remove('t')
stop.remove('s')
stop.remove('no')
stop.remove('not')
stop.remove('nor')


#### DATA PRE-PROCESSING ####
# defining a function that replaces multiple repeated characters with only two of the same characters -- e.g. "coool"
# becomes "cool"
def onlyone(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# defining the function that will pre-process the Tweets
def process_tweets(ourTweets):
    
    # Converting all the uppercase letters in the Tweets to lowercase
    ourTweets=ourTweets.lower()
    
    # Removing/stripping forward slashes, and speech marks
    ourTweets=ourTweets.strip('"/')
    
    # Removing hashtags
    ourTweets=ourTweets.replace("#","")
    
    # Removing the punctuations in the Tweets -- e.g. "3.99!!!?!" becomes "3.99"
    ourTweets=re.sub("(\?+)?\!+(\?+)?|(\?)+|(\,,+)|(\.)|(\,)|(\;)|(\w+:\/\/\S+)",'',ourTweets)
    
    # Replacing all the URLs from the Tweets with the word URL
    ourTweets=re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', ourTweets)
    
    # Removing all @usernames from our tweets as they are not informative of sentiment
    ourTweets=re.sub('@[^\s]+','',ourTweets)
    
    # Replacing all possible combinations of happy/positive emoticons with the string "hemoticons"
    ourTweets=re.sub('(\:[d])|(\:\))|(\(\:)|(\:(.)?\))|(\((.)?\:)',' hemoticons ', ourTweets)
    # Replacing all possible combinations of sad/negative emoticons with the string "semoticons"
    ourTweets=re.sub('(\:\()|(\)\:)|(\:(.)?\()|(\)(.)?\:)',' semoticons ',ourTweets)
    
    # Removing apostrophes from the Tweets
    ourTweets = ourTweets.replace("'", "")
    
    # Calling onlyone function to replace multiple repeated characters with a single character
    ourTweets=onlyone(ourTweets)
    
    # Changing plural words to singular by removing the extra "s" at the end
    s=".."
    for i in range(1,10):
        ourTweets=ourTweets.replace(s, " ")
        s=s+ ".+"
        
    # Splitting the string of Tweet into separate words
    words = ourTweets.split()

    # Adding words "URL" onto the stop set and removing them from the Tweets
    for i in words:
        if i=='URL':
            words.remove(i)
            stopwordList.append(i)
    
    # Removing the stopwords from the Tweets and joining the words into a string
    for i in words:
        if i in stop:
            words.remove(i)
    
    # Removing any blank spaces or extra brackets, which would be the result of previous processing
    for i in words:
        if i=='' or i==")" or i=="(":
            words.remove(i)
        
    return( " ".join( words ))


# Naming the thrid column of the training set, which holds the tweet string, 'tweet'
train['tweet']=train[3]

train['tweet']=train['tweet'].astype(str)


# Sets the row number in the training file in variable num_tweets
num_tweets = train[3].size

# Processing the training set Tweets by sending the data from text column of the file
# one Tweet is sent at a time for pre-processing
for i in range(len(train)):
    str1=train['tweet'][i]
    str2=process_tweets(str1)
    clean_train_tweets.append(str2)

# Pre-processing test tweets
test['tweet']=test[3]
for i in range(len(test)):
    str1=test['tweet'][i]
    str2=process_tweets(str1)
    clean_test_tweets.append(str2)

    
    
######   Building the feature matrix -- using CountVectorizer   ######

# Using CountVectorizer to build a features vector - specifying that the vector is made of word ngrams  
# rather than characters, removing the stopwords and initialising at max_features=5000.
vectorizer = CountVectorizer(analyzer = "word",  max_features = 5000) 
# Learning the vocabulary dictionary and returning term-document matrix
train_data_features = vectorizer.fit_transform(clean_tweets)
# Converts the feature matrix to an array
train_data_features = train_data_features.toarray()


# Learning the vocabulary dictionary and returning term-document matrix
test_data_features = vectorizer.transform(clean_test_tweets)
# Converts the feature matrix to an array
test_data_features = test_data_features.toarray()



#####  Building and training Random Forest classifier #####

# Initialising the Random Forest Classifier with 'n' estimators=
forest = RandomForestClassifier(n_estimators = 20) 
# Using feature vectors as our predictor, and the column that contains the tweet sentiments as our response variable
forest = forest.fit( train_data_features, train[2] )

# Predicting the labels based on model built from training data
result = forest.predict(test_data_features)
# Converting the results to data frame and combining id with labels
output = pd.DataFrame( data={"id":test[0], "sentiment":result} )
# Printing the Output
#print(output)



#### F1 Metric for evaluation ####

# Calculating the f1 score -- comparing mood predictions for 'dev-input-B' to the correct ones present in 'dev-gold-B'
# and tuning the parameters of the model accordingly to improve our results
y_true= pd.read_csv("twitter-dev-gold-B.tsv", header= None, sep="\t")
q = y_true[2]
f1_score(q , result,average='macro')


# In[28]:

############      SUBTASK B     ############

import nltk
import csv 
import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

# Sets/changes the directory to the document directory
os.chdir("/Users/Akhilesh Pandey/Desktop/art_corpus/ART_Corpus")
# Reads the training .tsv file and saves it as train
train = pd.read_csv("twitter-train-cleansed-B.tsv", header= None, sep="\t")
# Reads the testing/tuning .tsv file and saves it as test
test = pd.read_csv("twitter-dev-input-B.tsv", header=None, delimiter="\t", quoting=3 )

# Initialising the list to store the processed Tweets from the trainig data 
clean_train_tweets = []
# Initialising the list to store the processed Tweets from the test data
clean_test_tweets = []


# assigning the variable 'stop' to the set of english stopwords downloaded from the nltk package
stop=set(stopwords.words('english'))
# Removing stopwords that could be indicative of sentiment
stop.remove('t')
stop.remove('s')
stop.remove('no')
stop.remove('not')
stop.remove('nor')


#### DATA PRE-PROCESSING ####
# defining a function that replaces multiple repeated characters with only two of the same characters -- e.g. "coool"
# becomes "cool"
def onlyone(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# defining the function that will pre-process the Tweets
def process_tweets(ourTweets):
    
    # Converting all the uppercase letters in the Tweets to lowercase
    ourTweets=ourTweets.lower()
    
    # Removing/stripping forward slashes, and speech marks
    ourTweets=ourTweets.strip('"/')
    
    # Removing hashtags
    ourTweets=ourTweets.replace("#","")
    
    # Removing the punctuations in the Tweets -- e.g. "3.99!!!?!" becomes "3.99"
    ourTweets=re.sub("(\?+)?\!+(\?+)?|(\?)+|(\,,+)|(\.)|(\,)|(\;)|(\w+:\/\/\S+)",'',ourTweets)
    
    # Replacing all the URLs from the Tweets with the word URL
    ourTweets=re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', ourTweets)
    
    # Removing all @usernames from our tweets as they are not informative of sentiment
    ourTweets=re.sub('@[^\s]+','',ourTweets)
    
    # Replacing all possible combinations of happy/positive emoticons with the string "hemoticons"
    ourTweets=re.sub('(\:[d])|(\:\))|(\(\:)|(\:(.)?\))|(\((.)?\:)',' hemoticons ', ourTweets)
    # Replacing all possible combinations of sad/negative emoticons with the string "semoticons"
    ourTweets=re.sub('(\:\()|(\)\:)|(\:(.)?\()|(\)(.)?\:)',' semoticons ',ourTweets)
    
    # Removing apostrophes from the Tweets
    ourTweets = ourTweets.replace("'", "")
    
    # Calling onlyone function to replace multiple repeated characters with a single character
    ourTweets=onlyone(ourTweets)
    
    # Changing plural words to singular by removing the extra "s" at the end
    s=".."
    for i in range(1,10):
        ourTweets=ourTweets.replace(s, " ")
        s=s+ ".+"
        
    # Splitting the string of Tweet into separate words
    words = ourTweets.split()

    # Adding words "URL" onto the stop set and removing them from the Tweets
    for i in words:
        if i=='URL':
            words.remove(i)
            stopwordList.append(i)
    
    # Removing the stopwords from the Tweets and joining the words into a string
    for i in words:
        if i in stop:
            words.remove(i)
    
    # Removing any blank spaces or extra brackets, which would be the result of previous processing
    for i in words:
        if i=='' or i==")" or i=="(":
            words.remove(i)
        
    return( " ".join( words ))


# Naming the thrid column of the training set, which holds the tweet string, 'tweet'
train['tweet']=train[3]

train['tweet']=train['tweet'].astype(str)


# Sets the row number in the training file in variable num_tweets
num_tweets = train[3].size

# Processing the training set Tweets by sending the data from text column of the file
# one Tweet is sent at a time for pre-processing
for i in range(len(train)):
    str1=train['tweet'][i]
    str2=process_tweets(str1)
    clean_train_tweets.append(str2)

# Pre-processing test tweets
test['tweet']=test[3]
for i in range(len(test)):
    str1=test['tweet'][i]
    str2=process_tweets(str1)
    clean_test_tweets.append(str2)

    
    
######   Building the feature matrix -- using TfidfVectorizer   ######

# Using TFIDVectorizer to build a features vector - specifying that the vector is made of word ngrams  
# rather than characters, removing the stopwords and initialising at max_features=5000.
tf_vectorizer = TfidfVectorizer(analyzer = "word", ngram_range=(1,3), min_df=1, stop_words=stop, max_features=5000)
# Learning the vocabulary dictionary and returning term-document matrix
train_data_features = tf_vectorizer.fit_transform(clean_train_tweets)
# Converts the feature matrix to an array
train_data_features = train_data_features.toarray()


# Learning the vocabulary dictionary and returning term-document matrix
test_data_features = tf_vectorizer.transform(clean_test_tweets)
# Converts the feature matrix to an array
test_data_features = test_data_features.toarray()



#####  Building and training Random Forest classifier #####

# Initialising the Random Forest Classifier with 'n' estimators=
forest = RandomForestClassifier(n_estimators = 50) 
# Using feature vectors as our predictor, and the column that contains the tweet sentiments as our response variable
forest = forest.fit( train_data_features, train[2] )

# Predicting the labels based on model built from training data
result = forest.predict(test_data_features)
# Converting the results to data frame and combining id with labels
output = pd.DataFrame( data={"id":test[0], "sentiment":result} )
# Printing the Output
#print(output)



#### F1 Metric for evaluation ####

# Calculating the f1 score -- comparing mood predictions for 'dev-input-B' to the correct ones present in 'dev-gold-B'
# and tuning the parameters of the model accordingly to improve our results
y_true= pd.read_csv("twitter-dev-gold-B.tsv", header= None, sep="\t")
q = y_true[2]
f1_score(q , result,average='macro')


# In[30]:

##Testing on Test Data and prinitng the output in the Excel file
#We are using tf-idf as it is giving the highest accuracy in Subtask B
output = pd.DataFrame( data={"id":test[2], "sentiment":result} )
writer = pd.ExcelWriter('Output_Subtask_B.xlsx', engine='xlsxwriter')
output.to_excel(writer, sheet_name='Sheet1')
writer.save()

