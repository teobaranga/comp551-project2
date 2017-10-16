# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from math import log
import operator
#import numpy as np

x = pd.read_csv('dat_train_x_clean-ish.csv', encoding = 'utf-8') # Read csv 
# training file x data
y = pd.read_csv('dat_train_y_clean-ish.csv') # Read csv training file y data
# Split the data for Cross-Validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, 
                                                  random_state = 3)
x_train.Text = x_train.Text.str.lower() # Format all strings to lower case
x_val.Text = x_val.Text.str.lower() # Format all strings to lower case

Classes_Id = {}
for i in range(0,5):
    Classes_Id["Class_{0}_Id".format(i)] = y_train.loc[y_train['Category'] == i]
Classes_Text = {}
for Class in Classes_Id.keys():
    Id = Classes_Id[Class]['Id'].tolist()
    Classes_Text[Classes_Id[Class]['Category'].iloc[0]] = \
    x_train.loc[x_train['Id'].isin(Id)]
Doc_Count = {}
i = 0
for Class in Classes_Text.keys():
    Doc_Count[i] = len(Classes_Text[Class])
    i += 1
Total_doc_count = sum(Doc_Count.values())

# Calculate Prior Probabilities P(y=c)
priors = {}
for i in Doc_Count.keys():
    priors[i] = Doc_Count[i]/Total_doc_count

# Get Counts of letters in each sentence, in each class
for Class in Classes_Text:
    counts = []
    for doc in Classes_Text[Class]['Text'].tolist():
        cnt = Counter(str(doc))
        counts.append(cnt)
    Classes_Text[Class] = Classes_Text[Class].assign(Count = counts)
    
# Vocabulary is the same in each class. Make list of Vocab
Vocabulary = []
for count in Classes_Text[0]['Count']:
        for ltr in count:
            if ltr not in Vocabulary:
                Vocabulary.append(ltr)

# Formulate number of sentences that contain each letter
num_docs_with_ltr = {}
for Class in Classes_Text:
    for count in Classes_Text[Class]['Count']:
        for ltr in count:
            if ltr not in num_docs_with_ltr.keys():
                num_docs_with_ltr[ltr] = 1
            elif ltr in num_docs_with_ltr.keys():
                num_docs_with_ltr[ltr] += 1

# Get idf of each letter
idf = {}
for ltr in Vocabulary:
    idf[ltr] = log(Total_doc_count/(num_docs_with_ltr[ltr]+1))

# Get tfidf for each letter in each document, of each class
for Class in Classes_Text:
    tfidf = []
    for tf in Classes_Text[Class]['Count']:
        doc_tfidf = {}
        for ltr in tf:
            doc_tfidf[ltr] = (tf[ltr])*idf[ltr]
        tfidf.append(doc_tfidf)
    Classes_Text[Class] = Classes_Text[Class].assign(Tfidf = tfidf)

# Sum of tfidf of each letter in each class
Sum_of_Tfidf = {}
i = 0
for Class in Classes_Text:
    sum_of_tfidf_ltrs = dict.fromkeys(Vocabulary, 0)
    for ltr in Vocabulary:
        for tfidf in Classes_Text[Class]['Tfidf']:
            if ltr in tfidf.keys():
                sum_of_tfidf_ltrs[ltr] += tfidf[ltr]
    Sum_of_Tfidf[i] = sum_of_tfidf_ltrs
    i += 1
    
# Total sum of tfidf of all letters in each class
Total_sum_of_tfidf = {}
i = 0
for Class in Sum_of_Tfidf:
    Total_sum_of_tfidf[i] = sum(Sum_of_Tfidf[Class].values())
    i += 1

# Calculate Probabilities
PW_C = {}
i = 0
for Class in Sum_of_Tfidf:
    Pw_c = {}
    for ltr in Sum_of_Tfidf[Class].keys():
        Pw_c[ltr] = (Sum_of_Tfidf[Class][ltr] + 1)/(Total_sum_of_tfidf[Class] + 
            len(Vocabulary))
    PW_C[i] = Pw_c
    i += 1
        
## Applying NB Classifier
Prob_Classes = []
for docs in x_train.Text:
    doc = [lt for lt in str(docs) if lt.isalpha()]
    Scores = {}
    for Class in PW_C:
        Scores[Class] = log(priors[Class])
        for ltr in doc:
            Scores[Class] += log(PW_C[Class][ltr])
    Prob_Class = max(Scores.iteritems(), key = operator.itemgetter(1))[0]
    Prob_Classes.append(Prob_Class)
NB_Prob_Classes = y_train.assign(NB_Prob_Class = Prob_Classes)
print "Training Accuracy: %f" %(NB_Prob_Classes.Category == NB_Prob_Classes.NB_Prob_Class).mean()
cm = confusion_matrix(NB_Prob_Classes.Category, NB_Prob_Classes.NB_Prob_Class)
sns.heatmap(cm)
plt.show()


## NB Classifier Probabilities:
#P(word|class)=
#(word_count_in_class + 1)/(total_words_in_class + 
#  total_unique_words_in_all_classes(basically vocabulary of words in the entire training set))
#
#word_count_in_class : sum of(tf-idf_weights of the word for all the documents belonging to that class) 
#//basically replacing the counts with the tfidf weights of the same word calculated for every document within that class.
#
#total_words_in_class : sum of(tf-idf weights of all the words belonging to that class) 
#
#total_unique_words_in_all_classes : as is.
