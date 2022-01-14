#!/usr/bin/env python
# coding: utf-8

# From Given Data Set Buliding a Model to predict the survial using various algorithm 

# ### Author: Saurabh Nigam

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv('https://raw.githubusercontent.com/Premalatha-success/Supervised-Learning/main/titanic-training-data.csv')


# In[4]:


df.head(5)


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


sns.countplot(x="Survived",hue='Sex',data=df)


# In[8]:


sns.countplot(x="Parch",data=df)


# In[9]:


sns.countplot(x="Age",data=df)


# In[10]:


sns.heatmap(df.isnull(),yticklabels=False)


# In[11]:


df.drop('Cabin',axis=1,inplace=True)


# In[12]:


df.head()


# In[13]:


df.dropna(inplace=True)


# ### Converting some columns value to desried form using one hot Encoding

# In[14]:


Sex=pd.get_dummies(df['Sex'])


# In[15]:


Sex.head()


# In[16]:


Sex=pd.get_dummies(df['Sex'],drop_first=True)


# In[17]:


Sex.head()


# In[18]:


Embarked=pd.get_dummies(df['Embarked'])


# In[19]:


Embarked=pd.get_dummies(df['Embarked'],drop_first=True)


# In[20]:


Embarked.head()


# In[21]:


Pclass=pd.get_dummies(df['Pclass'])


# In[22]:


Pclass=pd.get_dummies(df['Pclass'],drop_first=True)


# In[23]:


Pclass.head()


# In[24]:


df=pd.concat([df,Pclass,Embarked,Sex],axis=1)


# In[25]:


df.drop(['Name','Sex','Ticket','Fare','Embarked'],axis=1,inplace=True)


# In[26]:


df.head()


# In[27]:


x=df.drop('Survived',axis=1)
y=df['Survived']


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)


# Fitting a model Using Logistic Regression

# In[29]:


model=LogisticRegression()
model.fit(x_train,y_train)


# Predicting the model

# In[30]:


pred=model.predict(x_test)


# In[31]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# In[32]:


from sklearn import metrics


# In[33]:


print(metrics.classification_report(y_test,pred))


# Confusion Metrics

# In[34]:


from sklearn.metrics import confusion_matrix


# In[35]:


confusion_matrix(y_test,pred)


# Decesion Tree Classifeir
# 

# In[36]:


from sklearn import tree


# In[37]:


from sklearn.tree import DecisionTreeClassifier


# we will Build a model Using Desciontreeclassifier using default criteria gini,other include Entropy 

# In[38]:


dtree = DecisionTreeClassifier(criterion='gini',random_state=2)
dtree.fit(x_train,y_train)


# ## Scoring our Decision Tree

# In[39]:


print(dtree.score(x_train,y_train))
print(dtree.score(x_test,y_test))


# ### Reducing over Fitting (Regularization)

# In[40]:


dtree = DecisionTreeClassifier(criterion='gini',max_depth=2,random_state=1)
dtree.fit(x_train,y_train)
print(dtree.score(x_train,y_train))
print(dtree.score(x_test,y_test))


# ### Ensemble Technique Bagging

# In[41]:


from sklearn.ensemble import BaggingClassifier


# In[42]:


bgcl= BaggingClassifier(n_estimators=50,base_estimator=dtree,random_state=1)
bgcl.fit(x_train,y_train)
y_pred=bgcl.predict(x_test)
print(bgcl.score(x_test,y_test))


# ### Ensemble Technique AdaBoosting

# In[43]:


from sklearn.ensemble import AdaBoostClassifier


# In[44]:


Adab= AdaBoostClassifier(n_estimators=5,base_estimator=dtree,random_state=1)
Adab.fit(x_train,y_train)
y_pred=Adab.predict(x_test)
print(Adab.score(x_test,y_test))


# ### Ensemble Technique Gradient Boosting

# In[45]:


from sklearn.ensemble import GradientBoostingClassifier


# In[46]:


Grad_B= GradientBoostingClassifier(n_estimators=20,random_state=1)
Grad_B.fit(x_train,y_train)
y_pred=Grad_B.predict(x_test)
print(Grad_B.score(x_test,y_test))


# In[ ]:




