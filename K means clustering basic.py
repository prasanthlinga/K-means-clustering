#!/usr/bin/env python
# coding: utf-8

# In[244]:


import numpy as np
import pandas as pd


# In[245]:


# generate our own dataset by using make_blobs function from scikit learn module
from sklearn.datasets import make_blobs

print(make_blobs)   # make_blobs function generate blobs of points with gaussioan distribution also called normal distribution 
#where maximum data falls near mean of the data 


# In[246]:


X,y =make_blobs(n_samples=1000,centers=4,cluster_std=0.6,n_features=4,random_state=42)

# cluster_std is standard deviation of clusters 
# no.of centers to generate   # no.of groups/clustres
# no.of features for each sample   #no.of columns


# In[247]:


print(X)


# In[248]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.scatter(X[:,1],X[:,2])


# In[249]:


plt.scatter(y[:],y[:])


# In[250]:


print(X.shape)


# In[251]:


print(y.shape)


# In[252]:


print(y)


# In[253]:


#y.reshape(1000,1)


# In[254]:


print(y.shape)


# In[255]:


from sklearn.cluster import KMeans
WCSS=[]  # within cluster sum of sqaures
for i in range(1,12):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=42)#kmeans++ :selects initial no.of clusters
    print(kmeans)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
print(WCSS)


# In[256]:


plt.figure(figsize=(10,10))
plt.plot(WCSS)  #if you provide only one parameter, matplotlib assumes it as a y parameter and automatically generates the X
plt.title("Elbow Method")
plt.xlabel("no.of clusters")
plt.ylabel("WCSS")
plt.show()


# In[271]:


# BY using elbow method, we get the no.of clusters for our data
#so no.of clusters =3
kmeans=KMeans(n_clusters=4,max_iter=300,n_init=10,random_state=42)
pred_y=kmeans.fit_predict(X)
pred_y


# In[272]:


a=pd.DataFrame(pred_y,columns=["cluster"])
#a["new"]=0
a


# In[273]:


b=a.groupby(["cluster"])


# In[274]:


print(b.first)


# In[275]:


a.value_counts()


# In[276]:


# just for practice
a["new"]=0
print(a)
a.value_counts()


# In[277]:


plt.figure(figsize=(10,10))
plt.scatter(X[:,1],X[:,2])
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],c='red',s=300)   # s is for size an d c is for colour


# In[ ]:




