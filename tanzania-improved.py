
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#loading the two datasets

#loading the train data
train_data = pd.read_csv('../input/training.csv')
print("The train data")
print(train_data.head())

#loading the test data
test_data = pd.read_csv('../input/test (1).csv')
print("The test data")
print(test_data.head())


# In[3]:


# check if the data has missing points with seaborn heatmap
import seaborn as sns
sns.heatmap(train_data.isnull(),yticklabels=False, cmap='viridis')


# In[4]:


# view the columns
train_data.columns


# In[5]:


# view the columns in  the test
x_test = test_data
x_test.columns


# In[6]:


#looking at the data above the X array has some features not in the x_test


# In[7]:


train_data.head(5)


# In[8]:


x_test.head()


# In[9]:


#dropping the labels that you are supposed to predict and the excess from train_head
cols = ['ID','mobile_money', 'savings', 'borrowing','insurance']
train_data = train_data.drop(cols, axis=1)
x_test = test_data.drop(['ID'], axis=1)


# In[10]:


#looking at the unique classification classes
train_data['mobile_money_classification'].unique()


# In[11]:


#lets look at both the train and test data and see if they match after the drop
train_data.columns


# In[12]:


x_test.columns


# In[13]:


X = train_data.drop(['mobile_money_classification'], axis=1)
y = train_data['mobile_money_classification']


# In[14]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42,n_estimators=1400,criterion='gini')
rf.fit(X, y)
#using the base model to build the feature importance
import pandas as pd
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)


# In[15]:


#lets drop the most irrelevant columns in both X and the x_test
#from the already done tests we find that dropping the last three is what works best
X1 = X.drop(['Q8_11','Q8_7','Q8_6'], axis=1)
x_test1 = x_test.drop(['Q8_11','Q8_7','Q8_6'], axis=1)


# In[16]:


#lets normalize the datasets
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X2 = scaler.fit_transform(X1)
X_test1=scaler.fit_transform(x_test1)


# In[17]:


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
scores = []
rf = RandomForestClassifier(random_state = 42,n_estimators=1400,criterion='gini')
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X2[train_index], X2[test_index], y[train_index], y[test_index]
    rf.fit(X_train, y_train)
    scores.append(rf.score(X_test, y_test))


# In[18]:


from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[19]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
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


# In[20]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf1 = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf1_random = RandomizedSearchCV(estimator = rf1, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[21]:


# Fit the random search model
#rf1_random.fit(X_train, y_train)


# In[22]:


#rf1_random.best_params_


# In[23]:


#grid search using cross validation
#Random search allowed us to narrow down the range for each hyperparameter. 
#Now that we know where to concentrate our search, we can explicitly specify every combination of settings to try.
#We do this with GridSearchCV, a method that, instead of sampling randomly from a distribution, evaluates all combinations we define. 
#To use Grid Search, we make another grid based on the best values provided by random search:
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [100, 100, 110, 120],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [800, 1000, 1600, 2000]
}
# Create a based model
rf2 = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf2, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
#grid_search.fit(X_train,y_train)
#grid_search.best_params_


# In[24]:


rf.fit(X_train, y_train)
scores.append(rf.score(X_test, y_test))


# In[25]:


scores.append(rf.score(X_test, y_test))


# In[26]:


print(np.mean(scores))


# In[27]:


from sklearn.ensemble import RandomForestClassifier
rfc2 = RandomForestClassifier(n_estimators=2000, max_depth=120, min_samples_split=10, 
                             min_samples_leaf=3,max_features='sqrt', bootstrap=True,random_state=42)
rfc2 .fit(X2, y)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier(n_estimators=1400, max_depth=100, min_samples_split=5, 
                             min_samples_leaf=4,max_features='sqrt', bootstrap=True,random_state=42)
rfc1 .fit(X2, y)


# In[29]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1400, max_depth=80, min_samples_split=10, 
                             min_samples_leaf=4,max_features='sqrt', bootstrap=True,random_state=42)
rfc .fit(X2, y)


# In[30]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

test_pred = rfc.predict_proba(X_test1)

test_pred = pd.DataFrame(rfc.predict_proba(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred = pd.DataFrame(data=q)
df_pred = df_pred[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[31]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

test_pred = rfc2.predict_proba(X_test1)

test_pred = pd.DataFrame(rfc2.predict_proba(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred2 = pd.DataFrame(data=q)
df_pred2 = df_pred[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[32]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

test_pred = rfc1.predict_proba(X_test1)

test_pred = pd.DataFrame(rfc1.predict_proba(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred1 = pd.DataFrame(data=q)
df_pred1 = df_pred[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[33]:


df_pred.head()


# In[34]:


df_pred1.head()


# In[35]:


#df_pred2.head()
df_pred1.round({"no_financial_services":4, "other_only":4, "mm_only":4, "mm_plus":4})


# In[36]:


df_pred1.to_csv('pred_set.csv', index=False) #save to csv file#


# FROM HERE THE CODE IS SPECIFIC TO XGBOOST MULTICLASS CLASSIFICATION

# In[37]:


#now since we have done some few possible things of trying to improve the model using
#random forest lets try using other algorithms (bring out the big guns)
# using xgboost
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.33)

X_train.shape, y_train.shape, X_test.shape, y_test.shape,X_test1.shape


# In[38]:


'''dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test)
dtest1 = xgb.DMatrix(data=X_test1)

params = {
    'max_depth': 4,
    'objective': 'multi:softmax',  # error evaluation for multiclass training
    'num_class': 4,
    'n_gpus': 0
}

#bst=xgb.train(params,X_train, y_train)
bst = xgb.train(params, dtrain)

pred = bst.predict(dtest)
print(pred)


#test_pred = bst.predict(dtest1)
print(test_pred)
print(labels.classes_)

params = {
    'max_depth': 4,
    'objective': 'multi:softmax',  # error evaluation for multiclass training
    'num_class': 4,
    'n_gpus': 0
}

#training_start = time.perf_counter()
#training_end = time.perf_counter()
#prediction_start = time.perf_counter()
preds = xgb.predict(X_test1)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))
'''


# In[39]:


import xgboost as xgb

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100,max_depth= 4,
    objective= 'multi:softmax',  
    num_class= 4,
    n_gpus= 0)
# error evaluation for multiclass training
xgb.fit(X_train, y_train)


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

test_pred = pd.DataFrame(xgb.predict(X_test1), columns=labels.classes_)
#test_pred = pd.DataFrame(bst.predict(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred1 = pd.DataFrame(data=q)
df_pred1 = df_pred[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[40]:


df_pred1.to_csv('pred_set.csv', index=False) #save to csv fil#

