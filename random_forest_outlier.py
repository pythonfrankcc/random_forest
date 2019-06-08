
# coding: utf-8

# In[ ]:


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


# In[ ]:


#importing the necessary dependencies
import matplotlib.pyplot as plt#for visualization


# In[ ]:


#let's now read the data
#loading the train data
train_data = pd.read_csv('../input/training.csv')
print("The train data")
train_data.head()


# In[ ]:


#loading the test data
test_data = pd.read_csv('../input/test (1).csv')
print("The test data")
test_data.head()


# In[ ]:


#information on the train_features
train_data.info()


# In[ ]:


#information on the test_features
test_data.info()


# In[ ]:


#from the above we can see that train_Data has more columns
# check if the data has missing points with seaborn heatmap
import seaborn as sns
sns.heatmap(train_data.isnull(),yticklabels=False, cmap='viridis')


# In[ ]:


#though unusual we can also look if the test data has some missing values
# check if the data has missing points with seaborn heatmap
import seaborn as sns
sns.heatmap(train_data.isnull(),yticklabels=False, cmap='viridis')


# In[ ]:


# view the columns in train_data and test_data
train_data.columns,test_data.columns


# In[ ]:


#lets check how thye submission csv is supposed to look like 
submission= pd.read_csv('../input/sample_submission (2).csv')
submission.head()


# In[ ]:


#dropping the labels that you are supposed to predict and the excess from train_head
cols = ['ID','mobile_money', 'savings', 'borrowing','insurance']
train_data = train_data.drop(cols, axis=1)
x_test = test_data.drop(['ID'], axis=1)


# In[ ]:


#you can try cleaning the data using z_score
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(train_data))
print(z)
train_data1 = train_data[(z < 3).all(axis=1)]


# In[ ]:


#look at the two dataframes and check the difference
print('train_data_shape:',train_data.shape)
print('train_data without outliers:',train_data1.shape)


# In[ ]:


#using the filtered data
X = train_data1.drop(['mobile_money_classification'], axis=1)
y = train_data1['mobile_money_classification']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42,n_estimators=1400,criterion='gini')
rf.fit(X, y)
#using the base model to build the feature importance
import pandas as pd
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)


# In[ ]:


#lets drop the most irrelevant columns in both X and the x_test atlest 4
#from the already done tests we find that dropping the last three is what works best
X1 = X.drop(['Q8_11','Q8_1','Q8_3','Q8_5','Q8_6','Q8_7'], axis=1)
x_test1 = x_test.drop(['Q8_11','Q8_1','Q8_3','Q8_5','Q8_6','Q8_7'], axis=1)


# In[ ]:


#making sure that you have dropped the columns
X1.columns,x_test1.columns


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
names=X1.columns
names1=x_test1.columns
scaler = MinMaxScaler(feature_range=(0, 1))
X2 = scaler.fit_transform(X1)
X3 = pd.DataFrame(X2, columns=names)
X_test=scaler.fit_transform(x_test1)
X_test1 = pd.DataFrame(X_test, columns=names1)


# In[ ]:


#making sure that you have dropped the columns
X3.columns,X_test1.columns


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X3, y, random_state=42, stratify=y)

X_train.shape, y_train.shape, X_test.shape, y_test.shape,


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc2 = RandomForestClassifier(n_estimators=2000, max_depth=120, min_samples_split=10, 
                             min_samples_leaf=3,max_features='sqrt', bootstrap=True,random_state=42)
rfc2 .fit(X_train,y_train)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

test_pred = rfc2.predict_proba(X_test1)

test_pred = pd.DataFrame(rfc2.predict_proba(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred = pd.DataFrame(data=q)
df_pred = df_pred[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[ ]:


df_pred.head()


# In[ ]:


df_pred.to_csv('pred_set.csv', index=False) #save to csv file#

