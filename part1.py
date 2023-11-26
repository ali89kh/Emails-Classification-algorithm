# 1- load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Train_split data
from sklearn.model_selection import train_test_split
# Encoding labels (emails' topics) (dependent features)
from sklearn.preprocessing import LabelEncoder
# text processing
import regex
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from string import punctuation
# Encoding independent features
from sklearn.feature_extraction.text import TfidfVectorizer
# training model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
# checking accuracy
from sklearn.metrics import confusion_matrix,accuracy_score

# 2-Data Preperation
# Get folders' names in the directory (equivalent to data labels)
path = r"directory to the data"
directories = os.listdir(path)
print(directories)
# load every email and put the features (emails' text) in x and the labels (emails' topics) in y
x=[]
y=[]
for directory in directories:
    path2=path+"/"+directory+"/"
    with os.scandir(path2) as entries:
        for entry in entries:
            with open(entry,"r", encoding="latin1") as file:
                m=file.read()
                if len(m) != 0:  # avoid empty emails (data quality issue)
                    x.append(m)
                    y.append(directory)

# take a look at the shapes of the inputs and check one sample
print('Input features shape : ', np.shape(x))
print('Labels shape : ', np.shape(y))
print('The label/topic of the first email is : ', y[0])
# print ('The email content is :\n', x[0], '\n\n\n')
# transform x from list to numpy array
x = np.array(x).reshape(-1)
print(x.shape)

# split data into train/test sets via a stratified fashion
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, shuffle= True, random_state=1200)

# Label encode the labels (0: , 1: ,2: ,3:)
le = LabelEncoder()
print('The existing unique email topics : ', pd.unique(y_train))
y_train = le.fit_transform(y_train)
y_test=le.transform(y_test)

# check the labels:

print('The resulted labels for each email topic : ', pd.unique(y_train))
print(len(y_train))

print(np.shape(X_train[y_train == 0]))
print(np.shape(X_train[y_train == 1]))
print(np.shape(X_train[y_train == 2]))
print(np.shape(X_train[y_train == 3]))

print(np.shape(X_test[y_test == 0]))
print(np.shape(X_test[y_test == 1]))
print(np.shape(X_test[y_test == 2]))
print(np.shape(X_test[y_test == 3]))
X_train_before=X_train.copy()


def text_cleaning_function(X_train):
    stop=stopwords.words('english')
    corpus=[]
    wordnet_lemmatizer=WordNetLemmatizer()

    for punct in punctuation:
        stop.append(punct)

    for text in X_train:
        sentences=WordPunctTokenizer().tokenize(text.lower())
        review=[regex.sub(u'\p{^Latin}', u'', w) for w in sentences if w.isalpha() and len(w) > 3]  #
        review=[wordnet_lemmatizer.lemmatize(w, pos="v") for w in review if not w in stop]
        review=' '.join(review)
        corpus.append(review)
    return corpus


# transform emails (cleaning ..etc)
X_train= text_cleaning_function(X_train_before)
# test the existence of empty emails
length_of_every_email = np.array([len(i) for i in X_train])
index_possible_empty_email = np.where(length_of_every_email == 0)[0]

# print the index of empty emails
print('Number of empty emails is : ', len(index_possible_empty_email))
# vectorized email texts (Tfid words --> numbers)
vectorizer_data = TfidfVectorizer()
X = vectorizer_data.fit_transform(X_train).toarray()
print(X.shape)
