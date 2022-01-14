#!/usr/bin/env python
# coding: utf-8

# ## **Linear Regression with Python Scikit Learn**
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# ### **Simple Linear Regression**
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# ### Author : Saurabh Nigam

# In[89]:


#importing all library required to solve the problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[90]:


#importing data 
Task_1=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[91]:


Task_1.head(10)


# #imported Data Sucessfully

# #Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# In[92]:


#plotting the distribution of scores
Task_1.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.xlabel('Percentage Scores')
plt.show()


# #From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# #Preparing the data
# 

# #The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[93]:


X=Task_1.drop('Scores',axis=1)
Y=Task_1['Scores']
X.head()


# In[94]:


Y.head()


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[95]:


from sklearn.model_selection import train_test_split


# In[96]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)


# ### Training Algorithm
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[97]:


from sklearn.linear_model import LinearRegression


# In[98]:


Lin_Reg= LinearRegression()
Lin_Reg.fit(X_train,Y_train)


# In[99]:


#plotting for test Data
plt.scatter(X,Y)
plt.show()


# ### **Making Predictions**
# Now that we have trained our algorithm, it's time to make some predictions.

# In[100]:


Y_pred=Lin_Reg.predict(X_test)


# In[101]:


#Comparing the data
df=pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})
df


# ### **Evaluating the model**
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[104]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred))


# In[ ]:




