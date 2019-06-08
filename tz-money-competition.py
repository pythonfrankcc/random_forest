
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#loading the train data
train_data = pd.read_csv('../input/training.csv')
print("The train data")
print(train_data.head())

#loading the test data
test_data = pd.read_csv('../input/test.csv')
print("The test data")
print(test_data.head())


# The data is already preprocessed so we'll not carry out data preprocessing...

# **Checking if the data has any missing values**

# In[3]:


# check if the data has missing points with seaborn heatmap
import seaborn as sns
sns.heatmap(train_data.isnull(),yticklabels=False, cmap='viridis')


# The data has no missing values.

# **Printing the list of unique classes**

# In[4]:


# view the columns
train_data.columns


# In[5]:


cols = ['ID','mobile_money', 'savings', 'borrowing','insurance']
train_data = train_data.drop(cols, axis=1)


# In[6]:


# the classes 
train_data['mobile_money_classification'].unique()


# There are 4 classes:
# * 0 - no mobile money and no other financial service (saving, borrowing, insurance)
# * 1 - no mobile money, but at least one other financial service
# * 2 - mobile money only
# * 3 - mobile money and at least one other financial service

# In[7]:


# checking the shape of the train data
train_data.shape


# **Training the Model**

# In[8]:


X = train_data.drop(['mobile_money_classification'], axis=1)
y = train_data['mobile_money_classification']

from sklearn.model_selection import train_test_split


#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)




# In[9]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
 #Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[10]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[11]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
#rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
#rf_random.fit(X_train,y_train)


# In[12]:


#rf_random.best_params_


# In[13]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=800, max_depth=10, min_samples_split=2, 
                             min_samples_leaf=4,max_features='sqrt', bootstrap=True,random_state=42)
rfc.fit(X_train, y_train)


# **PREDICTING**

# In[14]:


# viewing the shape of the test data
print("Shape of the test data:", test_data.shape)



# In[15]:



print("Columns in the test data:", test_data.columns)


# The test data has 4 less columns besides the target variable.

# In[16]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

x_test = test_data.drop(['ID'], axis=1)
test_pred = rfc.predict_proba(x_test)


# In[17]:


test_pred = pd.DataFrame(rfc.predict_proba(x_test)*1, columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred = pd.DataFrame(data=q)
df_pred = df_pred[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[18]:


df_pred.head()


# In[19]:


#df_pred.to_csv('pred_set8.csv', index=False) #save to csv file

#xgb_params = {'learning_rate': 0.05, 
 '''             'max_depth': 4,
              'subsample': 0.9,        
              'colsample_bytree': 0.9,
              'objective': 'binary:logistic',
              'silent': 1, 
              'n_estimators':100, 
              'gamma':1,         
              'min_child_weight':4}   
clf = xgb.XGBClassifier(**xgb_params, seed = 10)'''
#round(features.describe, 2)