
# Libraries import

import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Importing dataset

data = pd.read_csv('C:\\Users\\xxxx\\Desktop\\yelp_labelled.txt', sep='\t')

# Assigning column names to DataFrame

data.columns = ['Sentiment', 'Polarity']

# Splitting data into train and test set

X_train, X_test, y_train, y_test = train_test_split(data['Sentiment'], data['Polarity'], test_size=0.15)

# Removing stopwords, and useless text

removed_words = 0
stopwords_set = set(stopwords.words("english"))

# Removing punctuation marks

table = str.maketrans({key: None for key in string.punctuation})


def cleanTextData(X_train, y_train):
    words = []

    for sentence, sentiment in zip(X_train, y_train):
        #print(sentence)
        #print(sentiment)
        cleaned_words = []
        split_sentence = sentence.split()
        for word in split_sentence:
            if (word.lower() not in stopwords_set):
                cleaned_words.append((word.lower()).translate(table))
        words.append([cleaned_words, sentiment])
       # print(words)
    words_without_whitespace = []

    for a, b in words:
        #print(a) words[]
       # print(b)sentiment
        x = []
        for c in a:
            #print(c) word
            if (c not in '     ' and c.isdigit() == False and c not in stopwords_set):
                #print(c)
                x.append(c)
        if (x != []):
            #print(x)
            words_without_whitespace.append([x, b])
            #print(words_without_whitespace)
           # print("\n")
    return words_without_whitespace


words_without_whitespace = cleanTextData(X_train, y_train)
words_positive = []
words_negative = []


def frequencyOfWord(list_words):
    words_positive_dict = {}
    words_negative_dict = {}
    for value, sentiment in list_words:
        print(value)
        #print(sentiment)
        for word in value:
            if (sentiment == 1):
                words_positive.append(word)
            else:
                words_negative.append(word)
    for word in words_positive:
        try:
            words_positive_dict[word] += 1
        except:
            words_positive_dict[word] = 1
    for word in words_negative:
        try:
            words_negative_dict[word] += 1
        except:
            words_negative_dict[word] = 1
    return words_positive_dict, words_negative_dict


positive, negative = frequencyOfWord(words_without_whitespace)

# Prior probabilities
n_p = 0
n_n = 0
for sentiment in y_train:
    if (sentiment == 1):
        n_p += 1
    else:
        n_n += 1

p_positive = (n_p) / (n_p + n_n)
p_negative = (n_n) / (n_p + n_n)
words_included_pos = positive.keys()
words_included_neg = negative.keys()
w_features = []
w_features.extend(words_included_pos)
w_features.extend(words_included_neg)


def NaiveBayesClassification(test_data_X, test_data_y):
    cleanData = cleanTextData(test_data_X, test_data_y)
    probs = []
    for words, sentiment in cleanData:
        bag_of_words = set(words)
        pos_prob = 1
        neg_prob = 1
        for word in bag_of_words:
            try:
                pos_prob *= ((positive[word] + 1) / (sum(positive.values()) + 14))
            except:
                pos_prob *= (0 + 1) / (sum(positive.values()) + 14)
        pos_prob *= p_positive

        for word in bag_of_words:
            try:
                neg_prob *= ((negative[word] + 1) / (sum(negative.values()) + 14))
            except:
                neg_prob *= (0 + 1) / (sum(negative.values()) + 14)
        neg_prob *= p_negative

        probs.append([pos_prob, neg_prob])

    return probs


g = NaiveBayesClassification(X_test, y_test)

predictions = []

for x, y, z in zip(g, X_test, y_test):
    if (x[0] >= x[1]):
        predictions.append([1, y, z])
    else:
        predictions.append([0, y, z])

correct = 0
incorrect = 0
for x in predictions:
    if (x[0] == x[2]):
        correct += 1
    else:
        incorrect += 1

print("Result: Correct {} InCorrect {}".format(correct, incorrect))