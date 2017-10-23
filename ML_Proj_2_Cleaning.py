# -*- coding: utf-8 -*-
import pandas as pd
#import numpy as np

train_x = pd.read_csv('train_x_no_eng.csv', encoding = 'utf-8') # Read csv training file x data
train_y = pd.read_csv('train_y_no_eng.csv') # Read csv training file y data
print 'Length of original datasets\n x: %d y: %d' %(len(train_x),len(train_y))
NA_Id_list = train_x.loc[pd.isnull(train_x['Text']), 'Id'].tolist() # Get ids of empty elements
NA_idx = [] # Initialize list of nan indeces
# Loop through NA_Id_list and append indeces
for Id in NA_Id_list:
    idx = train_x.Id[train_x.Id == Id].index.tolist() # Get index of Id corresponding to nan element
    NA_idx.append(idx[0])
del idx # Clear variables
train_x = train_x.dropna(how='any') # Drop empty rows from x data
train_x = train_x.reset_index(drop=True)
train_y = train_y.drop(train_y.index[NA_idx]) # Drop labels corresponding to empty rows from y data
train_y = train_y.reset_index(drop=True)
train_x.Text = train_x.Text.str.lower() # Format all strings to lower case
train_x.Text = [list(txt) for txt in train_x.Text] # Split strings to list of letters
train_x_clean = [] # Initialize list of clean text
# Loop through each sentence and clean unwanted elements
for idx, txt in enumerate(train_x.Text):
        txt_no_num = [x for x in txt if x.isalpha()] # Only append non-digit elements
        txt_clean = [x for x in txt_no_num if x != u' '] # Only append strings != whitespace
        train_x_cl = {} # Initialize dictionary to use in rebuilding train_x dataframe
        train_x_cl['Id'] = train_x.Id[idx] # Dictionary Id item
        train_x_cl['Text'] = ''.join(txt_clean) # Dictionary Text item
        train_x_clean.append(train_x_cl)
#del txt_clean, train_x_cl, train_x, x # Clear variables
train_x = pd.DataFrame(train_x_clean) # Recreate dataframe
##
# I attempted to convert the pandas dataframe to a dictionary list but that is going to be huge and way to slow to work with
#txl = train_x.set_index('Id').T.to_dict('list') 
##
# 2nd round of cleaning empty cells that contained only digits
del NA_Id_list, NA_idx
NA_Id_list = train_x.loc[train_x['Text'] == u'', 'Id'].tolist() # Get ids of empty elements
NA_idx = [] # Initialize list of nan indeces
# Loop through NA_Id_list and append indeces
for Id in NA_Id_list:
    idx = train_x.Id[train_x.Id == Id].index.tolist() # Get index of Id corresponding to empty element
    NA_idx.append(idx[0])
train_x = train_x.drop(train_x.index[NA_idx]) # Drop empty rows from x data
train_x = train_x.reset_index(drop=True)
train_y = train_y.drop(train_y.index[NA_idx]) # Drop labels corresponding to empty rows from y data
train_y = train_y.reset_index(drop=True)
## Write new files
train_x.to_csv('train_x_clean-ish.csv', encoding = 'utf-8') # Write new csv file
train_y.to_csv('train_y_clean-ish.csv')
print 'Length of clean datasets\n x: %d y: %d' %(len(train_x),len(train_y))
