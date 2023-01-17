# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:02:43 2020

@author: listep
"""
import os
import numpy as np
import pandas as pd
import re
import tkinter as tk
from tkinter.filedialog import askopenfilename
import datetime as datetime

root = tk.Tk()
root.withdraw()
infile = askopenfilename(title='Select a data file',filetypes=[("Excel Files", "*.xlsx")]) #Read the file

os.chdir(os.path.dirname(infile))

data = pd.read_excel(infile)

columns = data.columns.tolist()

#Preprocessing
def process_strings(i):
    
    return str(i).strip().title()

for col in ['First Name', 'Last Name']:
    
    data[col] = data[col].apply(process_strings)
    

data.insert(2, 'Full Name', np.nan)

data['Full Name'] = data['First Name'] + ' ' + data['Last Name']

#Sanity Checks
test = data.groupby('Candidate ID').size()

test = test[test>1].index.tolist()

#Inspect
test = data[data['Candidate ID'] == 33712] #We probably want to remove this person from the incomplete

data.drop(203, inplace = True)

#Check NANs
test = data.isna().sum()

#test = data['Title'].value_counts(dropna = False)
#test = data['Other State'].value_counts(dropna = False)
#test = data['System Region Time Zone'].value_counts(dropna = False)
#test = data['Status'].value_counts(dropna = False)
#test = data['Group Name'].value_counts(dropna = False)
#test = data["GRAD State 'Other'"].value_counts(dropna = False)
#test = data["APP - Grad - Diversity: ATSI"].value_counts(dropna = False)
#test = data["Diversity: Disability"].value_counts(dropna = False)
#%%
#Include Merit List
root = tk.Tk()
root.withdraw()
infile = askopenfilename(title='Select a data file',filetypes=[("Excel Files", "*.xlsx")]) #Read the file

merit_list = pd.read_excel(infile, header = None)

#Remove brackets
subset = merit_list[0].apply(lambda i : '(' in str(i))  

merit_list.loc[subset, 0] = merit_list.loc[subset, 0].apply(lambda i : re.sub(r'\s\(\w+\)', '', i))

#Manipulate merit_list
merit_list['Successful Applicant'] = 'Yes'

merit_list.columns= ['Full Name','Successful Applicant']

merit_list['Full Name'] = merit_list['Full Name'].apply(process_strings)

#Merge
data = pd.merge(data, merit_list, how = 'outer', on = 'Full Name', indicator = True)

data['Successful Applicant'].value_counts(dropna = False)

#Fill
data['Successful Applicant'].fillna('No', inplace = True)

#%%
#Remove useless fields for analysis
useless_columns = ['Title', 'First Name', 'Last Name', 'Initial', 'Email', 'Email Address Confirmation', 'Other State', 'Mobile Number (for sms communication):',
                   'System Region Time Zone', 'Status', 'Job Flags', 'Global Flags', 'PS Act 1999 text', 'GRAD - Citizenship Status',
                   'Title.1', 'First Name.1', 'Last Name.1', 'APP - Preferred Name', 'Mobile Number (for sms communication):.1', 'Postal Address:',
                   "GRAD State 'Other'", "APP - Grad - Nearest Airport", "EMERGCONT", "Currently hold a security clearance:", "Clearance currently held:", 
                   "APP - Grad - Previously applied for ANAO Grad", "APP - Grad - Previously applied - Details", "Redundancy >12 Months", "Redundancy benefit date",
                   "Redundancy details", "Dismissed for misconduct", "Dismissed / Misconduct details", "What was the outcome", "Diversity: Statement", "App-Grad -Reasonable Adjustment Required",
                   "APP - Professional Affiliations?", "APP - Professional Affiliations Details", "Referee One Text Only", "Referee 1: Name", "Referee 1: Relationship", "Referee 1: Contact Number",
                   "Referee 1: Email address", "Referee Two Text Only", "Referee 2: Name", "Referee 2: Relationship", "Referee 2: Contact Number", "Referee 2: Email address", "Sensitivities Contacting Referees:",
                   "Sensitivities Contacting Referees - Details", "State.1", "Application Declaration", "Edit your application read only text", "_merge"]

data.drop(useless_columns, axis = 'columns', inplace = True)

#Cleaning Date Column
subset = data['APP - Grad - Qual Finish Date'].apply(lambda i : not((type(i) == datetime.datetime) | (type(i) == pd.Timestamp)))

uncleaned_dates = data[subset]['APP - Grad - Qual Finish Date']

test = pd.to_datetime(uncleaned_dates.apply(lambda i : str(i)), errors = 'coerce')

subset_nan = pd.isna(test)

#First phase of filling
data['APP - Grad - Qual Finish Date'] = data['APP - Grad - Qual Finish Date'].apply(lambda i : pd.to_datetime(str(i), errors= 'coerce'))

#Second phase
nan_values = uncleaned_dates[subset_nan]

def clean_dates(i):
    
    if re.findall(r'\w+\s\d{2,4}', str(i)):
        return re.findall(r'\w+\s\d{2,4}', str(i))[0]
    
    else:
        return np.nan
    
nan_values = nan_values.apply(clean_dates)

test = pd.to_datetime(nan_values.apply(lambda i : str(i)), errors = 'coerce')

subset = ~pd.isna(test)

data.loc[subset.index.tolist(), 'APP - Grad - Qual Finish Date'] = test[subset]

data.to_csv('Cleaned Data.csv', index = False)
#%%
#Miscellaneous

#Text Analysis
import nltk
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk.tokenize import wordpunct_tokenize
import matplotlib.pyplot as plt

text_series = data['APP - Grad - Personal Achievements'].dropna()

text_list = text_series.str.cat(sep = ' ')

stop_word = set(ENGLISH_STOP_WORDS).union(('s', 'australia', 'australian', 'monash', 'university', 'grammar'))

list_of_words = [i.lower() for i in wordpunct_tokenize(text_list) if i.lower() not in stop_word and i.isalpha()]

#Visualising using base tokenizer without lemmitizisation
wordfreqdist = nltk.FreqDist(list_of_words)

mostcommon = wordfreqdist.most_common(30)

#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()
#%%
#Stemming words
'''
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

stemmed_list_of_words = [stemmer.stem(i) for i in list_of_words]
'''

#Lemmatize words
from nltk.stem import WordNetLemmatizer #Lemmatizes words
from nltk.corpus import wordnet #install manually first, create corpora
from nltk.tokenize import punkt #install manually first, create tokenizer
from nltk.tag import perceptron #install manually first, create taggers
from nltk import pos_tag

wnl = WordNetLemmatizer()

lemmatized_list_of_words = [wnl.lemmatize(i) for i in list_of_words]

wordfreqdist = nltk.FreqDist(lemmatized_list_of_words)

mostcommon = wordfreqdist.most_common(30)

#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()

#5.1 (Key topics that graduates highlight in their personal achievements -> types of people)

#Using POS tagging
final_lemmatized_list_of_words = []
for word, tag in pos_tag(lemmatized_list_of_words):
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    if not wntag:
        lemma = word
    else:
        lemma = wnl.lemmatize(word, wntag)
        
    final_lemmatized_list_of_words.append(lemma)   

wordfreqdist = nltk.FreqDist(final_lemmatized_list_of_words)

mostcommon = wordfreqdist.most_common(30)


#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()


#%%
#2 gram analysis
from nltk import bigrams, trigrams

eng_bigrams = bigrams(final_lemmatized_list_of_words)

bigrams_list_of_words = [ (w1, w2) for w1, w2 in eng_bigrams if len(w1) >=5 and len(w2) >= 5 ]

bigrams_list_of_words = [tuple(sorted(i)) for i in bigrams_list_of_words]

bigrams_list_of_words = [' '.join(i) for i in bigrams_list_of_words]

wordfreqdist = nltk.FreqDist(bigrams_list_of_words)

mostcommon = wordfreqdist.most_common(20)

#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()


#Tri gram analysis
eng_bigrams = trigrams(final_lemmatized_list_of_words)

trigrams_list_of_words = [ (w1, w2, w3) for w1, w2, w3 in eng_bigrams if len(w1) >=2 and len(w2) >= 2 and len(w3) >= 2]

wordfreqdist = nltk.FreqDist(trigrams_list_of_words)

mostcommon = wordfreqdist.most_common(40)

#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()

#%%
#Analysis of successful candidates

success_data = data[data['Successful Applicant'] == 'Yes']

text_series = success_data['APP - Grad - Personal Achievements'].dropna()

text_list = text_series.str.cat(sep = ' ')

stop_word = set(ENGLISH_STOP_WORDS).union(('s', 'university'))

list_of_words = [i.lower() for i in wordpunct_tokenize(text_list) if i.lower() not in stop_word and i.isalpha()]

#Visualising using base tokenizer without lemmitizisation
wordfreqdist = nltk.FreqDist(list_of_words)

mostcommon = wordfreqdist.most_common(30)

#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()

#Lemmatization
wnl = WordNetLemmatizer()

lemmatized_list_of_words = [wnl.lemmatize(i) for i in list_of_words]

wordfreqdist = nltk.FreqDist(lemmatized_list_of_words)

mostcommon = wordfreqdist.most_common(30)

#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()

#5.1 (Key topics that graduates highlight in their personal achievements -> types of people)

#Using POS tagging
final_lemmatized_list_of_words = []
for word, tag in pos_tag(lemmatized_list_of_words):
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    if not wntag:
        lemma = word
    else:
        lemma = wnl.lemmatize(word, wntag)
        
    final_lemmatized_list_of_words.append(lemma)   

wordfreqdist = nltk.FreqDist(final_lemmatized_list_of_words)

mostcommon = wordfreqdist.most_common(30)


#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()


#Visualising bigrams
eng_bigrams = bigrams(final_lemmatized_list_of_words)

bigrams_list_of_words = [ (w1, w2) for w1, w2 in eng_bigrams if len(w1) >=5 and len(w2) >= 5 ]

bigrams_list_of_words = [tuple(sorted(i)) for i in bigrams_list_of_words]

bigrams_list_of_words = [' '.join(i) for i in bigrams_list_of_words]

wordfreqdist = nltk.FreqDist(bigrams_list_of_words)

mostcommon = wordfreqdist.most_common(20)

#Visualise
plt.figure(figsize=(20,10))
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()

#%%
#Build a model
corpus = data['APP - Grad - Personal Achievements'].dropna()

def lemmatize(i):
    
    list_of_words = i.split()
    
    wnl = WordNetLemmatizer()
    
    final_lemmatized_list_of_words = []
    
    for word, tag in pos_tag(list_of_words):
        
        wntag = tag[0].lower()
        
#        if wntag in ['np', 'nnp']: #Tag propoer nouns and nouns
#            
#            continue
        
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        
        if not wntag:
            
            lemma = wnl.lemmatize(word)
            
        else:
            
            lemma = wnl.lemmatize(word, wntag)
        
        final_lemmatized_list_of_words.append(lemma)  
        
    return ' '.join(final_lemmatized_list_of_words)

corpus = corpus.apply(lemmatize)

#Split Data
data['Result'] = np.where(data['Successful Applicant'] == 'Yes', 1, 0)

data['Result'] = data['Result'].astype('category')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(corpus, data[~pd.isna(data['APP - Grad - Personal Achievements'])][['Result']], test_size = 0.33, random_state = 40)

#Create CountVectorizer object
vectorizer = CountVectorizer(ngram_range = (1,2), min_df = 2, stop_words = 'english') #Fitting a vectorizer for our model
                        
#Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

#Transform X_test
X_test_bow = vectorizer.transform(X_test)

#Building a classifier (Naive Bayes)
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

clf = MultinomialNB(alpha = 0.15).fit(X_train_bow, np.ravel(np.array(y_train)))

predicted = clf.predict(X_test_bow)

print(metrics.accuracy_score(y_test, predicted))

metrics.confusion_matrix(y_test, predicted)

print(metrics.classification_report(y_test, predicted))
##############################################################################
#Interpretting Important Features
feature_names = vectorizer.get_feature_names()

coefs_with_fns = sorted(zip(np.exp(clf.feature_log_prob_[0]), feature_names))

top = zip(coefs_with_fns[:20], coefs_with_fns[:-(20+1):-1])

for (coef_1, fn1), (coef_2, fn2) in top:
    
    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn1, coef_2, fn2))

##############################################################################
#Test prediction
statement = "I am a bad candidate."
prediction = clf.predict(vectorizer.transform([statement]))[0]
print("Candidate predicted by the classifier is %i" % (prediction))


#TDIDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tdidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), stop_words = 'english', max_features = 1000)

#Fit and transform X_train
X_train_bow = tdidf_vectorizer.fit_transform(X_train)

#Transform X_test
X_test_bow = tdidf_vectorizer.transform(X_test)

clf = MultinomialNB(alpha = 0.15).fit(X_train_bow, np.ravel(np.array(y_train)))

predicted = clf.predict(X_test_bow)

print(metrics.accuracy_score(y_test, predicted))

metrics.confusion_matrix(y_test, predicted)

print(metrics.classification_report(y_test, predicted))

#Random Sampling.

#%%
#Model Building
#Let's gridsearch to tune parameters
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 

from sklearn.ensemble import GradientBoostingClassifier
#Doing a simple fit, fixing the random_state for reproducibility
estimator = GradientBoostingClassifier(random_state = 0, verbose = 1)
estimator.fit(X_train_bow, y_train)

print("Accuracy on training set: {:.3f}".format(estimator.score(X_train_bow, y_train)))

print("Accuracy on test set: {:.3f}".format(estimator.score(X_test_bow, y_test)))


param_grid = {'max_depth': [1, 2, 3, 4, 5],
                'n_estimators': [50, 75, 100, 200],
                'learning_rate': [0.01, 0.1, 0.25 , 0.5, 0.75, 1]}

#Tuning Process
gridsearch = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
gridsearch.fit(X_train_bow, y_train)



#%%



#Analysis

#4.1.1 Education

#Number of applicants applying within each degree discipline


