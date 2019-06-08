
# coding: utf-8

# In[192]:


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


# In[193]:


#removing unnecessary
import warnings
warnings.filterwarnings('ignore')
#loading the train data
train_data = pd.read_csv('../input/training.csv')
print("The train data")
print(train_data.head())

#loading the test data
test_data = pd.read_csv('../input/test (1).csv')
print("The test data")
print(test_data.head())


# In[194]:


# check if the data has missing points with seaborn heatmap
import seaborn as sns
sns.heatmap(train_data.isnull(),yticklabels=False, cmap='viridis')


# In[195]:


# view the columns
train_data.columns


# In[196]:


cols = ['ID','mobile_money', 'savings', 'borrowing','insurance']
train_data = train_data.drop(cols, axis=1)
x_test = test_data.drop(['ID'], axis=1)


# In[197]:


# the classes 
train_data['mobile_money_classification'].unique()


# so we can classify the data into 4 classes

# In[198]:


X = train_data.drop(['mobile_money_classification'], axis=1)
y = train_data['mobile_money_classification']


# In[199]:


#names=train_data['feauture_names']


# In[200]:


x_test.head(5)


# In[201]:


#lets normalize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X_test1=scaler.fit_transform(x_test)


# In[202]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = XGBClassifier(objective ='multi:softprob', colsample_bytree = 0.3, learning_rate = 0.1,
               max_depth = 10, alpha = 10, gamma=0.2, n_estimators = 320)
scores1 = []
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    xg_reg.fit(X_train, y_train)
    scores1.append(xg_reg.score(X_test, y_test))

#xg_reg.fit(X,y)

#preds = xg_reg.predict(X_test)
#rmse = np.sqrt(mean_squared_error(y_test, preds))
#print("RMSE: %f" % (rmse))


# In[203]:


import xgboost as xgb
from xgboost import XGBClassifier
xg_reg1 = XGBClassifier(objective ='multi:softprob', colsample_bytree = 0.3, learning_rate = 0.1,
               max_depth = 10, alpha = 10, gamma=0.2, n_estimators = 320)
xg_reg1.fit(X,y,eval_metric='auc')


# In[204]:


print('Start Predicting')
predictions=xg_reg1.predict(X_test)
pred_proba=xg_reg1.predict_proba(X_test)[:,1]
print('score:%.4g'% metrics.accuracy_score(y_test,predictions))


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
scores = []
rf = RandomForestClassifier(random_state = 42,n_estimators=320)
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    rf.fit(X_train, y_train)
    scores.append(rf.score(X_test, y_test))


# In[ ]:


xg_reg.fit(X_train, y_train)
scores1.append(xg_reg.score(X_test, y_test))


# In[ ]:


rf.fit(X_train, y_train)
scores.append(rf.score(X_test, y_test))


# In[ ]:


print(np.mean(scores1))


# In[ ]:


print(np.mean(scores))


# In[ ]:


xg_reg.fit(X_train, y_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1400, max_depth=80, min_samples_split=5, 
                             min_samples_leaf=4,max_features='sqrt', bootstrap=True,random_state=42)
rfc .fit(X_train, y_train)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

#x_test = test_data.drop(['ID'], axis=1)
#test_pred = rfc.predict_proba(X_test1)
test_pred = xg_reg.predict(X_test1)


# In[17]:


test_pred = pd.DataFrame(xg_reg.predict_proba(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred = pd.DataFrame(data=q)
df_pred = df_pred[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]



# In[ ]:


df_pred.head()


# In[ ]:


df_pred.to_csv('pred_set.csv', index=False) #save to csv fil
train.pop(‘Cabin’)
train.pop(‘Name’)
train.pop(‘Ticket’)
train.shape
> (891, 9)

# In[ ]:


#df_pred.head()


# In[ ]:




