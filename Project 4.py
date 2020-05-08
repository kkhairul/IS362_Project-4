#!/usr/bin/env python
# coding: utf-8

# # IS 362 – Project 4

# ### KHAIRUL CHOWDHURY

#  • Start with the mushroom data in the pandas DataFrame that you constructed in your “Assignment – Preprocessing Data with sci-kit learn.”
# 
# • Use scikit-learn to determine which of the two predictor columns that you selected (odor and one other column of your choice) most accurately predicts whether or not a mushroom is poisonous. There is an additional challenge here—to use scikit-learn’s predictive classifiers, you’ll want to convert each of your two (numeric categorical) predictor columns into a set of columns. See for one approach pandas get_dummies() method.
# 
# • Clearly state your conclusions along with any recommendations for further analysis.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import sklearn.model_selection
import sklearn.linear_model
from sklearn import metrics


# ##### The next step is to create our variables, select the columns we will be using, read the data using pandas and displaying our data using the head() function.

# In[3]:


url          = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
columns      = [0,4,5,22]
column_names = ['Class', 'Bruises', 'Odor', 'Habitat']
mushrooms    = pd.read_csv(url, sep=',', usecols=columns, header=None, names=column_names)
mushrooms.head()


# In[5]:


mushrooms['Class']   = mushrooms['Class'].map({'e':0, 'p':1})
mushrooms['Bruises'] = mushrooms['Bruises'].map({'t':10, 'f':11})
mushrooms['Odor']    = mushrooms['Odor'].map({'a':20, 'l':21, 'c':22, 'y':23, 'f':24, 'm':25, 'n':26, 'p':27, 's':28})
mushrooms['Habitat'] = mushrooms['Habitat'].map({'g':30, 'l':31, 'm':32, 'p':33, 'u':34, 'w':35, 'd':36})
mushrooms.head()


# In[6]:


bruises = pd.get_dummies(pd.Series(mushrooms['Bruises']))
odor    = pd.get_dummies(pd.Series(mushrooms['Odor']))
habitat = pd.get_dummies(pd.Series(mushrooms['Habitat']))


# In[7]:


mushrooms_new = pd.concat([mushrooms['Class'], bruises, odor, habitat], axis=1)
columns = list(mushrooms_new.iloc[:, :-1])
mushrooms_new.head(10)


# In[8]:


X = mushrooms_new.iloc[:, :-1].values
Y = mushrooms_new.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=1)


# In[9]:


linreg = sklearn.linear_model.LinearRegression()
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)
t = [1, 0]
p = [1, 0]

print(sklearn.metrics.mean_absolute_error(t, p))
print(sklearn.metrics.mean_squared_error(t, p))
print(np.sqrt(sklearn.metrics.mean_squared_error(t, p)))


# ###### We will now run our formula on all the data to see what result we achieve.

# In[10]:


X = mushrooms_new.iloc[:, :-7].values
Y = mushrooms_new.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=1)
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)

print(sklearn.metrics.mean_absolute_error(Y_test, Y_pred))


# ##### We will now remove the Odor data to see what result we achieve.

# In[11]:


X = mushrooms_new.iloc[:, 4:12].values
Y = mushrooms_new.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=1)
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)

print(sklearn.metrics.mean_absolute_error(Y_test, Y_pred))


# ##### We will now remove the Bruises data to see what result we achieve.

# In[12]:


X = mushrooms_new.iloc[:, 2:4].values
Y = mushrooms_new.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=1)
linreg.fit(X_train, Y_train)
Y_pred = linreg.predict(X_test)

print(sklearn.metrics.mean_absolute_error(Y_test, Y_pred))


# ## Conclusion

# As we can see from the results, the Bruises category has no effect on predicting anything as when you remove that data, the results are the same
# When we remove the Habitat data, we get a much better number at roughly 1.77
# 
# We get the best result when we remove the Odor data. Here we achieved a result of roughly .2675
# 
# What this data tells us is that if we would like to predict which mushrooms are poisonous or not, our best chance would be to look at the Habitat data we are provided as it is the best indicator as per our results.

# In[ ]:




