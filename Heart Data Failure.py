#!/usr/bin/env python
# coding: utf-8

#     From Given Heart Data set Build a model using various Algorithm and find out which model is best for given data set

# ### Author- Saurabh Nigam

# In[120]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[121]:


Heart_data= pd.read_csv(r'C:\Users\Dhruv Singh\Desktop\ptyhon ass\Heart Failure data.csv')


# ### Analyizing the Data

# In[122]:


Heart_data.head(15)


# In[123]:


Heart_data.info


# In[124]:


Heart_data.isnull().sum()


# In[125]:


Heart_data.head(5)


# ### Cleaning the Data by eliminating columns which does not affect the result

# In[126]:


Heart_data.drop(['creatinine_phosphokinase','serum_creatinine','serum_sodium'],axis=1,inplace=True)


# In[127]:


Heart_data.head()


# ### spliting the data into test and train with taking DEATH_EVENT as target variable

# In[128]:


x=Heart_data.drop('DEATH_EVENT',axis=1)
y=Heart_data['DEATH_EVENT']


# In[129]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=1)


# ## Fitting a model Using Loigistic Regression

# In[130]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[131]:


model=LogisticRegression()
model.fit(x_train,y_train)


# Predicting a model

# In[132]:


pred=model.predict(x_test)


# In[133]:


accuracy_score(y_test,pred)


# In[134]:


from sklearn import metrics


# In[135]:


print(metrics.classification_report(y_test,pred))


# ## using Decision Tree

# In[136]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[137]:


dtree=DecisionTreeClassifier(criterion='gini',max_depth=2,random_state=2)
dtree.fit(x_train,y_train)


# ### Scoring our tree

# In[138]:


print(dtree.score(x_train,y_train))
print(dtree.score(x_test,y_test))


# ## Using GradeintBoosting

# In[139]:


from sklearn.ensemble import GradientBoostingClassifier


# In[140]:


gbcl=GradientBoostingClassifier(n_estimators=20,random_state=1)
gbcl.fit(x_train,y_train)
y_pred=gbcl.predict(x_test)
print(gbcl.score(x_test,y_test))


# ## Using Ada boost

# In[143]:


from sklearn.ensemble import AdaBoostClassifier


# In[149]:


Adab=AdaBoostClassifier(n_estimators=20,base_estimator=dtree,random_state=2)
Adab.fit(x_train,y_train)
Y_pred=Adab.predict(x_test)
print(Adab.score(x_test,y_test))


# In[ ]:




