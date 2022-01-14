#!/usr/bin/env python
# coding: utf-8

# From the Given Data set predict the future prices of house using linear regression

# ### Author: Saurabh Nigam

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv(r'C:\Users\Dhruv Singh\Desktop\ptyhon ass\House data.csv')


# In[3]:


df.head(10)


# Cleaning the Data

# In[4]:


df.drop(['view','waterfront','id','yr_built','yr_renovated','zipcode','lat'],inplace=True,axis=1)


# In[5]:


df.drop('date',inplace=True,axis=1)


# In[6]:


df.head()


# In[7]:


df.corr()


# ## Spliting the Data

# In[8]:


x=df.drop('price',axis=1)
y=df[['price']]


# In[9]:


x.head()


# In[10]:


y.head()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=1)


# In[13]:


print('feature set size',x.shape)
print('feature set size',y.shape)


# In[14]:


print('train size',x_train.shape)
print('train size',y_train.shape)


# In[15]:


print('test set size of x:',x_test.shape)
print('test set size of y:',y_test.shape)


# In[16]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Using the Linerregreesion

# In[17]:


lm = LinearRegression()


# Fitting the Model

# In[18]:


lm.fit(x_train,y_train)


# In[19]:


print('The intercept of linear model:',lm.intercept_)


# In[20]:


print('The Coefficient of linear model:',lm.coef_)


# In[21]:


test_pred=lm.predict(x_test)


# In[22]:


train_pred=lm.predict(x_train)


# In[23]:


metrics.mean_squared_error(y_test,test_pred)


# In[24]:


metrics.mean_absolute_error(y_train,train_pred)


# In[25]:


np.sqrt(metrics.mean_squared_error(y_train,train_pred))


# In[26]:


prediction=lm.predict(x_test)


# In[27]:


print('type of predcit object',type(prediction))


# In[28]:


print('The metrics mean error',metrics.mean_squared_error(y_test,prediction))


# In[29]:


print('The sqaured mean  error',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[30]:


print('The r square error',round(metrics.r2_score(y_test,prediction)))


# In[ ]:




