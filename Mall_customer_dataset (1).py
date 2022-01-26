#!/usr/bin/env python
# coding: utf-8

# ### We have to analyze the spending score basis on income, age etc. Factors Using K-means 

# ### Author : Saurabh Nigam

# In[9]:


#import essential libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sklearn


# In[12]:


# import the dataset and slice the important features

dataset = pd.read_csv(r'C:\Users\Dhruv Singh\Desktop\ptyhon ass\Mall_Customers (4).csv')
dataset.head()#visualizing the Data


# In[13]:


X = dataset.iloc[:, [3, 4]].values


# In[5]:


#We have to find the optimal K value for clustering the data. 
#Now we are using the Elbow method to find the optimal K value.
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11): 
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  kmeans.fit(X) 
  wcss.append(kmeans.inertia_)


# “init” argument is the method for initializing the centroid. We calculated the WCSS value for each K value. Now we have to plot the WCSS with K value

# In[6]:


plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


# The point at which the elbow shape is created is 5, that is, our K value or an optimal number of clusters is 5. Now let’s train the model on the dataset with a number of clusters 5.

# In[7]:


kmeans = KMeans(n_clusters = 5, init = "k-means++", random_state = 42)
y_kmeans = kmeans.fit_predict(X)
y_kmeans


# In[8]:


#y_kmeans give us different clusters corresponding to X. Now let’s plot all the clusters using matplotlib.

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 60, c = 'yellow', label = 'Cluster5') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.legend() 

plt.show()

