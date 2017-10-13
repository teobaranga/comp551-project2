# -*- coding: utf-8 -*-
import pandas as pd
import timeit
from langdetect import detect_langs
start = timeit.default_timer()
train_x = pd.read_csv('dat_train_x.csv', encoding = 'utf-8') # Read csv training file x data
train_y = pd.read_csv('dat_train_y.csv') # Read csv training file y data
EN_list = [] # Initialize list of english sent. indeces
for idx, txt in enumerate(train_x.Text):
    try:
        test = detect_langs(txt) # Detect language
        for item in test:
            if item.lang == u'en' and item.prob > 0.8: # Only if language is english with a probability > 0.8
                EN_list.append(idx) # Add sent. index to list
    except Exception as LangDetectException: 
        continue # Ignore any elements that do not contain enough features to identify language
    except KeyboardInterrupt:
        break
train_x = train_x.drop(train_x.index[EN_list]) # Drop elements corresponding to indeces in list
train_y = train_y.drop(train_y.index[EN_list])
train_x.to_csv('dat_train_x_no_eng.csv', encoding = 'utf-8') # Write new csv file
train_y.to_csv('dat_train_y_no_eng.csv')
stop = timeit.default_timer()
print stop - start